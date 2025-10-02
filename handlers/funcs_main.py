from aiogram import Router, types, F
from aiogram.types import InputFile
import io
import time
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np
import pandas as pd
from model.prod import analyze
from aiogram.types import BufferedInputFile

router = Router()

MODEL_PATH = "model/prod/models/best.pt"
DEVICE = 'cpu'
CONF_THRESH = 0.2
IOU_THRESH = 0.5
MAX_PHOTOS = 10

# Загружаем модель один раз
model = YOLO(MODEL_PATH)


async def crop_by_boxes(image: Image.Image, result) -> list:
    """
    Возвращает список обрезанных изображений по найденным коробкам YOLO.
    """
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    crops = []
    for (x1, y1, x2, y2) in boxes:
        cropped = image.crop((x1, y1, x2, y2))
        arr = np.array(cropped, dtype=np.uint8)

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        crops.append(arr)
    return crops


async def detecting_trees_on_image_bytes(file_bytes: bytes):
    """
    Обработка одного изображения через YOLO + analyze из байтов.
    Возвращает PIL.Image с нарисованными боксами и DataFrame с результатами.
    """
    img = Image.open(io.BytesIO(file_bytes))
    np_img = np.array(img)

    results = model.predict(
        source=np_img,
        imgsz=640,
        device=DEVICE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        max_det=50,
        save=False,
        stream=False,
        verbose=False
    )
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return img, pd.DataFrame(), 0

    # Обрезаем деревья
    cropped_images = await crop_by_boxes(img, result)
    analysis_results = analyze(cropped_images, DEVICE)

    # Создаем DataFrame
    results_df = pd.DataFrame(analysis_results)
    results_df['tree_n'] = results_df.reset_index()['index'] + 1
    results_df = results_df.rename(columns={
        'species': 'Вид',
        'mechanical': 'Мех.Поврежд.',
        'hollow': 'Дупла',
        'rot': 'Ств.Гниль',
        'cracks': 'Трещины',
        'em_slope': 'Оп. накл.',
        'kappa': 'Каппы',
        'tree_n': 'Номер'
    })
    results_df = results_df[['Номер', 'Вид', 'Мех.Поврежд.', 'Дупла',
                             'Ств.Гниль', 'Трещины', 'Оп. накл.', 'Каппы']]

    # Рисуем боксы и нумерацию
    draw = ImageDraw.Draw(img)
    boxes = result.boxes.xyxy.cpu().numpy()
    try:
        font = ImageFont.truetype("arial.ttf", 80)
    except:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        draw.rectangle([x1, y1, x2, y2], outline="red", width=6)
        draw.text(
            (cx, cy),
            str(i),
            font=font,
            fill="black",
            stroke_width=3,
            stroke_fill="white"
        )

    return img, results_df, len(boxes)


@router.message(lambda message: message.text == "Загрузить одно или несколько фото")
async def ask_for_photos(message: types.Message):
    await message.answer("Пришлите фото (можно несколько).")


@router.message(F.content_type == types.ContentType.PHOTO)
async def handle_photos(message: types.Message):
    photos = message.photo

    if not photos:
        await message.answer("Нет фото для обработки!")
        return

    # Отправляем сообщение о принятии фото в работу
    await message.answer("Фотография принята в работу!")

    # Обрабатываем только самое большое фото (последнее в списке)
    # или можно выбрать фото с наибольшим размером
    if len(photos) > 1:
        # Берем самое большое фото (с наибольшим file_size или последнее в списке)
        largest_photo = photos[-1]  # Последнее фото обычно самое большое
    else:
        largest_photo = photos[0]

    total_trees = 0
    start_time = time.time()

    # Обрабатываем только одно фото
    file = await message.bot.get_file(largest_photo.file_id)
    file_bytes = await message.bot.download_file(file.file_path)
    img_bytes = file_bytes.read()

    img_with_boxes, results_df, trees_count = await detecting_trees_on_image_bytes(img_bytes)
    total_trees += trees_count

    # Отправка обработанного изображения
    bio = io.BytesIO()
    img_with_boxes.save(bio, format="JPEG")
    bio.seek(0)
    await message.answer_photo(photo=BufferedInputFile(bio.getvalue(), filename="result.jpg"))

    # Отправка результатов для каждого дерева отдельным сообщением
    if not results_df.empty:
        for idx, row in results_df.iterrows():
            tree_details = []
            for column_name, value in row.items():
                tree_details.append(f"<b>{column_name}</b>: {value}")

            tree_message = "\n".join(tree_details)
            await message.answer(f"Информация о дереве #{idx + 1}:\n{tree_message}")
    else:
        await message.answer("Деревья не обнаружены на этом фото.")

    elapsed_time = time.time() - start_time
    await message.answer(
        f"Обработка завершена!\nНайдено деревьев: {total_trees}\nВремя обработки: {elapsed_time:.2f} секунд."
    )
