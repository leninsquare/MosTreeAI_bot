from aiogram.types import KeyboardButton, ReplyKeyboardMarkup
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode

kb = [
        [KeyboardButton(text="Загрузить одно или несколько фото")]
    ]
keyboard = ReplyKeyboardMarkup(
        keyboard=kb,
        resize_keyboard=True,
        input_field_placeholder="Выберете действие"
    )

router = Router()

@router.message(Command("start"))
async def start_handler(msg: Message):
    text = (
        "Я бот проекта MosTreeAI команды 'Рога и копыта'. Пришли мне фотографии, и я определю дерево и типы его повреждений! "
        "Я могу обработать как одну, так и несколько фотографий. В качестве результата я верну изображение с пронумерованными "
        "детектированными деревьями и таблицу с информацией. "
        "Пока проект на стадии разработки — не отправляй больше 10 фотографий за раз, иначе бот может зависнуть."
    )
    await msg.answer(text.format(), reply_markup=keyboard, parse_mode=ParseMode.HTML)