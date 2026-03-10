"""
Zera Auto Bot — Telegram bot for car listing posts.
Supports ICE (gasoline/diesel) and Electric Vehicles (EV).
Uses GPT-4o vision to analyze car photos and generate posts.
"""

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    PhotoSize,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config file path
# ---------------------------------------------------------------------------

CONFIG_FILE = Path("bot_config.json")

# ---------------------------------------------------------------------------
# FSM States
# ---------------------------------------------------------------------------


class ConfigStates(StatesGroup):
    waiting_telegram_token = State()
    waiting_openai_key = State()
    waiting_channel_id = State()
    confirm_settings = State()


# ---------------------------------------------------------------------------
# BotConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class BotConfig:
    telegram_token: str = ""
    openai_api_key: str = ""
    channel_id: str = ""
    manager_handle: str = "@zera_mgmt"
    rub_bonus: int = 120_000

    def save(self) -> None:
        CONFIG_FILE.write_text(
            json.dumps(
                {
                    "telegram_token": self.telegram_token,
                    "openai_api_key": self.openai_api_key,
                    "channel_id": self.channel_id,
                    "manager_handle": self.manager_handle,
                    "rub_bonus": self.rub_bonus,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> "BotConfig":
        if not CONFIG_FILE.exists():
            return cls()
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return cls(**data)
        except Exception as exc:
            logger.warning("Cannot read config: %s", exc)
            return cls()

    def is_configured(self) -> bool:
        return bool(self.telegram_token and self.openai_api_key)


# ---------------------------------------------------------------------------
# Constants — emoji
# ---------------------------------------------------------------------------

EMOJI_CAR = "🚗"
EMOJI_LIGHTNING = "⚡"
EMOJI_BATTERY = "🔋"
EMOJI_SPEED = "🏎"
EMOJI_RANGE = "📍"
EMOJI_FIRE = "🔥"
EMOJI_STAR = "⭐"
EMOJI_MONEY = "💰"
EMOJI_PHONE = "📞"
EMOJI_CHECK = "✅"
EMOJI_ARROW = "➡️"
EMOJI_WRENCH = "🔧"
EMOJI_GLOBE = "🌍"
EMOJI_SPARKLES = "✨"
EMOJI_CROWN = "👑"
EMOJI_ROCKET = "🚀"

# ---------------------------------------------------------------------------
# Constants — GPT prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ANALYZER = """Ты — эксперт-оценщик автомобилей. Твоя задача — проанализировать фотографии автомобиля и вернуть строго JSON-ответ.

Правила:
1. Определи марку, модель, год выпуска, комплектацию.
2. Определи тип автомобиля: "electric" если это электромобиль, иначе "ice" (ДВС).
3. Для электромобиля определи: мощность двигателя (л.с.), ёмкость батареи (кВт·ч), запас хода (км), тип кузова.
4. Для ДВС: объём двигателя (например "1.5T"), мощность (л.с.), тип кузова, тип топлива.
5. Если на фото есть VIN-номер или номерной знак — установи has_vin=true.
6. Придумай уникальный слоган (1–2 предложения) для рекламного поста.
7. Определи примерную рыночную цену в рублях (только число, без пробелов и символов).
8. Перечисли ключевые особенности (до 5 пунктов).

Верни ТОЛЬКО валидный JSON без markdown-блоков в следующем формате:
{
  "vehicle_type": "ice" | "electric",
  "make": "...",
  "model": "...",
  "year": 2024,
  "trim": "...",
  "body_type": "...",
  "engine": "...",
  "power_hp": 150,
  "fuel_type": "...",
  "battery_kwh": null,
  "range_km": null,
  "price_rub": 2500000,
  "slogan": "...",
  "features": ["...", "..."],
  "has_vin": false
}
Для электромобилей engine = null, fuel_type = null; battery_kwh и range_km обязательны.
Для ДВС battery_kwh = null, range_km = null.
"""

SLOGAN_STYLE_EV = (
    "Слоган должен подчёркивать экологичность, инновации и свободу движения без АЗС."
)
SLOGAN_STYLE_ICE = (
    "Слоган должен подчёркивать динамику, надёжность и удовольствие от вождения."
)

# ---------------------------------------------------------------------------
# UTF-16 entity helpers
# ---------------------------------------------------------------------------


def utf16_len(text: str) -> int:
    """Return the length of *text* measured in UTF-16 code units."""
    return len(text.encode("utf-16-le")) // 2


def build_entity(etype: str, offset: int, length: int) -> Dict[str, Any]:
    return {"type": etype, "offset": offset, "length": length}


def utf16_offset(text: str, char_offset: int) -> int:
    """Convert a character offset to a UTF-16 code-unit offset."""
    return utf16_len(text[:char_offset])


# ---------------------------------------------------------------------------
# CarInfo dataclass
# ---------------------------------------------------------------------------


@dataclass
class CarInfo:
    vehicle_type: str = "ice"  # "ice" | "electric"
    make: str = ""
    model: str = ""
    year: int = 0
    trim: str = ""
    body_type: str = ""
    # ICE fields
    engine: Optional[str] = None
    power_hp: Optional[int] = None
    fuel_type: Optional[str] = None
    # EV fields
    battery_kwh: Optional[float] = None
    range_km: Optional[int] = None
    # Common
    price_rub: int = 0
    slogan: str = ""
    features: List[str] = field(default_factory=list)
    has_vin: bool = False

    @property
    def is_electric(self) -> bool:
        return self.vehicle_type == "electric"

    def display_price(self, addition: int) -> str:
        total = self.price_rub + addition
        # Format with spaces as thousands separator
        return f"{total:,}".replace(",", " ") + " ₽"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CarInfo":
        info = cls()
        info.vehicle_type = data.get("vehicle_type", "ice")
        info.make = data.get("make", "")
        info.model = data.get("model", "")
        info.year = int(data.get("year") or 0)
        info.trim = data.get("trim", "")
        info.body_type = data.get("body_type", "")
        info.engine = data.get("engine")
        raw_hp = data.get("power_hp")
        info.power_hp = int(raw_hp) if raw_hp is not None else None
        info.fuel_type = data.get("fuel_type")
        raw_bat = data.get("battery_kwh")
        info.battery_kwh = float(raw_bat) if raw_bat is not None else None
        raw_range = data.get("range_km")
        info.range_km = int(raw_range) if raw_range is not None else None
        raw_price = data.get("price_rub", 0)
        info.price_rub = int(raw_price) if raw_price else 0
        info.slogan = data.get("slogan", "")
        info.features = list(data.get("features") or [])
        info.has_vin = bool(data.get("has_vin", False))
        return info


# ---------------------------------------------------------------------------
# ChatGPT Analyzer
# ---------------------------------------------------------------------------


class ChatGPTAnalyzer:
    OPENAI_URL = "https://api.openai.com/v1/chat/completions"
    MODEL = "gpt-4o"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def _make_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _encode_image(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    async def analyze_photos(
        self,
        photos_bytes: List[bytes],
        extra_hint: str = "",
    ) -> Optional[CarInfo]:
        """Send photos to GPT-4o and return parsed CarInfo or None on failure."""
        if not photos_bytes:
            return None

        system_text = SYSTEM_PROMPT_ANALYZER
        if extra_hint:
            system_text += "\n\nДополнительная информация от пользователя: " + extra_hint

        content: List[Dict[str, Any]] = [{"type": "text", "text": system_text}]

        # GPT-4o supports up to 4 high-detail images per request without hitting
        # context-length limits while keeping API costs reasonable.
        for img_bytes in photos_bytes[:4]:
            b64 = self._encode_image(img_bytes)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high",
                    },
                }
            )

        payload = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1024,
            "temperature": 0.2,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.OPENAI_URL,
                    headers=self._make_headers(),
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error("OpenAI API error %d: %s", resp.status, body[:300])
                        return None
                    data = await resp.json()
        except asyncio.TimeoutError:
            logger.error("OpenAI API timeout")
            return None
        except aiohttp.ClientError as exc:
            logger.error("OpenAI API request error: %s", exc)
            return None

        try:
            raw_text = data["choices"][0]["message"]["content"]
            # Strip markdown code fences if present
            raw_text = re.sub(r"```(?:json)?", "", raw_text).strip().strip("`").strip()
            parsed = json.loads(raw_text)
            return CarInfo.from_dict(parsed)
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse GPT response: %s", exc)
            logger.debug("Raw GPT response: %s", data)
            return None


# ---------------------------------------------------------------------------
# Text formatting helpers
# ---------------------------------------------------------------------------


def _feature_lines(features: List[str]) -> str:
    return "\n".join(f"  {EMOJI_CHECK} {f}" for f in features) if features else ""


def format_post_ev(car: CarInfo, price_addition: int, contact: str) -> str:
    """Format a Telegram post for an electric vehicle."""
    title = f"{EMOJI_LIGHTNING} {car.year} {car.make} {car.model}"
    if car.trim:
        title += f" {car.trim}"

    lines = [
        title,
        "",
        f"{EMOJI_SPARKLES} {car.slogan}",
        "",
        f"{EMOJI_BATTERY} Батарея: {car.battery_kwh} кВт·ч" if car.battery_kwh else "",
        f"{EMOJI_RANGE} Запас хода: {car.range_km} км" if car.range_km else "",
        f"{EMOJI_SPEED} Мощность: {car.power_hp} л.с." if car.power_hp else "",
        f"{EMOJI_CAR} Кузов: {car.body_type}" if car.body_type else "",
        "",
    ]

    if car.features:
        lines.append(f"{EMOJI_CROWN} Особенности:")
        lines.append(_feature_lines(car.features))
        lines.append("")

    lines.append(f"{EMOJI_FIRE} Цена: {car.display_price(price_addition)}")

    if contact:
        lines.append("")
        lines.append(f"{EMOJI_PHONE} Контакт: {contact}")

    return "\n".join(line for line in lines if line is not None)


def format_post_ice(car: CarInfo, price_addition: int, contact: str) -> str:
    """Format a Telegram post for an ICE vehicle."""
    title = f"{EMOJI_CAR} {car.year} {car.make} {car.model}"
    if car.trim:
        title += f" {car.trim}"

    engine_str = ""
    if car.engine:
        engine_str = car.engine
        if car.fuel_type:
            engine_str += f" ({car.fuel_type})"

    lines = [
        title,
        "",
        f"{EMOJI_SPARKLES} {car.slogan}",
        "",
        f"{EMOJI_WRENCH} Двигатель: {engine_str}" if engine_str else "",
        f"{EMOJI_SPEED} Мощность: {car.power_hp} л.с." if car.power_hp else "",
        f"{EMOJI_CAR} Кузов: {car.body_type}" if car.body_type else "",
        "",
    ]

    if car.features:
        lines.append(f"{EMOJI_STAR} Особенности:")
        lines.append(_feature_lines(car.features))
        lines.append("")

    lines.append(f"{EMOJI_MONEY} Цена: {car.display_price(price_addition)}")

    if contact:
        lines.append("")
        lines.append(f"{EMOJI_PHONE} Контакт: {contact}")

    return "\n".join(line for line in lines if line is not None)


def format_post(car: CarInfo, price_addition: int, contact: str) -> str:
    """Dispatch to EV or ICE formatter based on vehicle type."""
    if car.is_electric:
        return format_post_ev(car, price_addition, contact)
    return format_post_ice(car, price_addition, contact)


def detect_ev_keywords(text: str) -> bool:
    """Return True if text mentions EV-specific keywords."""
    keywords = [
        "электромобиль",
        "kwh",
        "кВт·ч",
        "запас хода",
        "батарея",
        "электро",
        "bev",
        "ev",
    ]
    lower = text.lower()
    # Use word-boundary-aware check for short tokens like "ev"
    return any(
        re.search(r"\b" + re.escape(kw) + r"\b", lower) if len(kw) <= 3 else kw in lower
        for kw in keywords
    )


# ---------------------------------------------------------------------------
# Album (media group) storage
# ---------------------------------------------------------------------------


class AlbumStorage:
    """Collects photos from a media group before processing."""

    def __init__(self, delay: float = 1.5) -> None:
        self._albums: Dict[str, Dict[str, Any]] = {}
        self._delay = delay

    def add(self, media_group_id: str, message: Message) -> None:
        if media_group_id not in self._albums:
            self._albums[media_group_id] = {
                "messages": [],
                "timestamp": time.monotonic(),
                "chat_id": message.chat.id,
                "extra_text": "",
            }
        self._albums[media_group_id]["messages"].append(message)

    def is_complete(self, media_group_id: str) -> bool:
        if media_group_id not in self._albums:
            return False
        elapsed = time.monotonic() - self._albums[media_group_id]["timestamp"]
        return elapsed >= self._delay

    def get(self, media_group_id: str) -> Optional[Dict[str, Any]]:
        return self._albums.get(media_group_id)

    def pop(self, media_group_id: str) -> Optional[Dict[str, Any]]:
        return self._albums.pop(media_group_id, None)

    def set_text(self, media_group_id: str, text: str) -> None:
        if media_group_id in self._albums:
            self._albums[media_group_id]["extra_text"] = text


album_storage = AlbumStorage()

# ---------------------------------------------------------------------------
# Global config & router
# ---------------------------------------------------------------------------

config: BotConfig = BotConfig.load()
router = Router()

# ---------------------------------------------------------------------------
# Utility: download photo bytes
# ---------------------------------------------------------------------------


async def download_photo(bot: Bot, photo: PhotoSize) -> Optional[bytes]:
    """Download photo from Telegram and return raw bytes."""
    try:
        file = await bot.get_file(photo.file_id)
        if file.file_path is None:
            return None
        buffer = await bot.download_file(file.file_path)
        if buffer is None:
            return None
        return buffer.read()
    except Exception as exc:
        logger.error("Failed to download photo: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Handler: /start
# ---------------------------------------------------------------------------


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(
        f"{EMOJI_ROCKET} Привет! Я бот для создания авторекламных постов.\n\n"
        f"Отправьте мне фото автомобиля (одно или несколько), и я создам пост "
        f"для вашего канала.\n\n"
        f"Команды:\n"
        f"/setup — настройка бота\n"
        f"/status — текущие настройки\n"
        f"/help — справка"
    )


# ---------------------------------------------------------------------------
# Handler: /help
# ---------------------------------------------------------------------------


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        f"{EMOJI_STAR} Справка по боту\n\n"
        f"1. Отправьте одно или несколько фото автомобиля.\n"
        f"2. Можно добавить подпись к фото с деталями (год, пробег, цена).\n"
        f"3. Бот проанализирует фото и создаст пост для канала.\n"
        f"4. Поддерживаются как обычные автомобили, так и электромобили.\n\n"
        f"{EMOJI_LIGHTNING} Электромобили определяются автоматически!\n\n"
        f"Настройки: /setup\n"
        f"Текущий статус: /status"
    )


# ---------------------------------------------------------------------------
# Handler: /status
# ---------------------------------------------------------------------------


@router.message(Command("status"))
async def cmd_status(message: Message) -> None:
    cfg = config
    lines = [
        f"{EMOJI_WRENCH} Текущие настройки:",
        "",
        f"Канал: {cfg.channel_id or 'не задан'}",
        f"Надбавка к цене: +{cfg.rub_bonus:,} ₽".replace(",", " "),
        f"Менеджер: {cfg.manager_handle or 'не задан'}",
        f"OpenAI ключ: {'✅ задан' if cfg.openai_api_key else '❌ не задан'}",
    ]
    await message.answer("\n".join(lines))


# ---------------------------------------------------------------------------
# Handler: /setup  (FSM)
# ---------------------------------------------------------------------------


@router.message(Command("setup"))
async def cmd_setup(message: Message, state: FSMContext) -> None:
    await state.set_state(ConfigStates.waiting_telegram_token)
    await message.answer(
        f"{EMOJI_WRENCH} Настройка бота.\n\n"
        f"Шаг 1/3: Введите Telegram Bot Token (получить у @BotFather):\n"
        f"(Отправьте /skip чтобы оставить текущее значение)"
    )


@router.message(StateFilter(ConfigStates.waiting_telegram_token))
async def setup_telegram_token(message: Message, state: FSMContext) -> None:
    if message.text and message.text.strip() != "/skip":
        config.telegram_token = message.text.strip()
        config.save()
    await state.set_state(ConfigStates.waiting_openai_key)
    await message.answer(
        f"Шаг 2/3: Введите ваш OpenAI API ключ (sk-...):\n"
        f"(Отправьте /skip чтобы оставить текущее значение)"
    )


@router.message(StateFilter(ConfigStates.waiting_openai_key))
async def setup_openai_key(message: Message, state: FSMContext) -> None:
    if message.text and message.text.strip() != "/skip":
        config.openai_api_key = message.text.strip()
        config.save()
    await state.set_state(ConfigStates.waiting_channel_id)
    await message.answer(
        f"Шаг 3/3: Введите ID или username канала для публикации\n"
        f"(например: @my_channel или -1001234567890)\n"
        f"(Отправьте /skip чтобы оставить текущее значение)"
    )


@router.message(StateFilter(ConfigStates.waiting_channel_id))
async def setup_channel_id(message: Message, state: FSMContext) -> None:
    if message.text and message.text.strip() != "/skip":
        config.channel_id = message.text.strip()
        config.save()
    await state.clear()
    await message.answer(
        f"{EMOJI_CHECK} Настройки сохранены!\n\n"
        f"Канал: {config.channel_id or 'не задан'}\n"
        f"Менеджер: {config.manager_handle}\n"
        f"OpenAI: {'✅' if config.openai_api_key else '❌'}"
    )


# ---------------------------------------------------------------------------
# Inline keyboard for post confirmation
# ---------------------------------------------------------------------------


def make_confirm_keyboard(media_group_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"{EMOJI_CHECK} Опубликовать",
                    callback_data=f"publish:{media_group_id}",
                ),
                InlineKeyboardButton(
                    text="❌ Отмена",
                    callback_data=f"cancel:{media_group_id}",
                ),
            ]
        ]
    )


# ---------------------------------------------------------------------------
# Pending posts storage
# ---------------------------------------------------------------------------

# Maps a key -> {"text": str, "photo_ids": List[str], "car": CarInfo}
pending_posts: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Core processing: analyze photos and send preview
# ---------------------------------------------------------------------------


async def process_photos(
    bot: Bot,
    chat_id: int,
    photo_ids: List[str],
    extra_text: str = "",
) -> None:
    """Download photos, analyze with GPT-4o, send post preview to user."""
    if not config.openai_api_key:
        await bot.send_message(
            chat_id,
            f"{EMOJI_WRENCH} OpenAI API ключ не настроен. Используйте /setup",
        )
        return

    # Download all photos
    status_msg = await bot.send_message(chat_id, "🔍 Анализирую фотографии...")

    photos_bytes: List[bytes] = []
    for file_id in photo_ids:
        # Get the PhotoSize-like object
        try:
            file = await bot.get_file(file_id)
            if file.file_path:
                buf = await bot.download_file(file.file_path)
                if buf:
                    data = buf.read()
                    if data:
                        photos_bytes.append(data)
        except Exception as exc:
            logger.error("Error downloading file %s: %s", file_id, exc)

    if not photos_bytes:
        await status_msg.edit_text("❌ Не удалось загрузить фотографии.")
        return

    # Analyze with GPT
    extra_hint = extra_text
    if detect_ev_keywords(extra_text):
        extra_hint += " [ЭЛЕКТРОМОБИЛЬ]"

    analyzer = ChatGPTAnalyzer(config.openai_api_key)
    car = await analyzer.analyze_photos(photos_bytes, extra_hint=extra_hint)

    if car is None:
        await status_msg.edit_text(
            "❌ Не удалось проанализировать фото. Попробуйте ещё раз или другое фото."
        )
        return

    if car.has_vin:
        await status_msg.edit_text(
            f"⚠️ На фото обнаружен VIN-номер или гос. номер.\n"
            f"По соображениям конфиденциальности публикация заблокирована.\n"
            f"Пожалуйста, сделайте фото без регистрационных данных."
        )
        return

    # Generate post text
    post_text = format_post(car, config.rub_bonus, config.manager_handle)

    # Store pending post
    post_key = f"{chat_id}_{int(time.monotonic() * 1000)}"
    pending_posts[post_key] = {
        "text": post_text,
        "photo_ids": photo_ids,
        "car": car,
    }

    # Send preview
    vehicle_label = f"{EMOJI_LIGHTNING} Электромобиль" if car.is_electric else f"{EMOJI_CAR} Автомобиль"
    preview_header = (
        f"{vehicle_label} распознан: {car.year} {car.make} {car.model}\n"
        f"Комплектация: {car.trim or 'неизвестна'}\n\n"
        f"Предварительный просмотр поста:"
    )

    await status_msg.edit_text(preview_header)

    # Send post text as preview
    if len(photo_ids) == 1:
        await bot.send_photo(
            chat_id=chat_id,
            photo=photo_ids[0],
            caption=post_text,
            reply_markup=make_confirm_keyboard(post_key),
        )
    else:
        from aiogram.types import InputMediaPhoto

        media = [
            InputMediaPhoto(media=fid, caption=post_text if i == 0 else None)
            for i, fid in enumerate(photo_ids[:10])
        ]
        await bot.send_media_group(chat_id=chat_id, media=media)
        await bot.send_message(
            chat_id=chat_id,
            text="Подтвердите публикацию:",
            reply_markup=make_confirm_keyboard(post_key),
        )


# ---------------------------------------------------------------------------
# Handler: incoming photos
# ---------------------------------------------------------------------------


@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot) -> None:
    if not message.photo:
        return

    # Get the best quality photo
    best_photo = message.photo[-1]
    extra_text = message.caption or ""

    if message.media_group_id:
        # Part of an album — collect all, then process after delay
        album_storage.add(message.media_group_id, message)
        if extra_text:
            album_storage.set_text(message.media_group_id, extra_text)

        # Schedule processing after the album collection window
        await asyncio.sleep(album_storage._delay)

        album = album_storage.get(message.media_group_id)
        if album is None:
            return  # already processed

        # Only the first message in the group triggers processing
        if album["messages"][0].message_id != message.message_id:
            return

        album_data = album_storage.pop(message.media_group_id)
        if album_data is None:
            return

        photo_ids = [msg.photo[-1].file_id for msg in album_data["messages"] if msg.photo]
        await process_photos(
            bot=bot,
            chat_id=message.chat.id,
            photo_ids=photo_ids,
            extra_text=album_data["extra_text"],
        )
    else:
        await process_photos(
            bot=bot,
            chat_id=message.chat.id,
            photo_ids=[best_photo.file_id],
            extra_text=extra_text,
        )


# ---------------------------------------------------------------------------
# Callback: publish / cancel
# ---------------------------------------------------------------------------


from aiogram.types import CallbackQuery


@router.callback_query(F.data.startswith("publish:"))
async def callback_publish(callback: CallbackQuery, bot: Bot) -> None:
    post_key = callback.data.split(":", 1)[1]
    post = pending_posts.pop(post_key, None)

    if post is None:
        await callback.answer("Пост уже опубликован или отменён.", show_alert=True)
        return

    if not config.channel_id:
        await callback.answer("Канал не настроен! Используйте /setup", show_alert=True)
        return

    photo_ids: List[str] = post["photo_ids"]
    post_text: str = post["text"]

    try:
        if len(photo_ids) == 1:
            await bot.send_photo(
                chat_id=config.channel_id,
                photo=photo_ids[0],
                caption=post_text,
            )
        else:
            from aiogram.types import InputMediaPhoto

            media = [
                InputMediaPhoto(media=fid, caption=post_text if i == 0 else None)
                for i, fid in enumerate(photo_ids[:10])
            ]
            await bot.send_media_group(chat_id=config.channel_id, media=media)

        await callback.answer(f"{EMOJI_CHECK} Опубликовано!", show_alert=False)
        if callback.message:
            await callback.message.edit_reply_markup(reply_markup=None)
            await callback.message.reply(f"{EMOJI_CHECK} Пост успешно опубликован в {config.channel_id}!")
    except Exception as exc:
        logger.error("Publish error: %s", exc)
        await callback.answer(f"Ошибка публикации: {exc}", show_alert=True)


@router.callback_query(F.data.startswith("cancel:"))
async def callback_cancel(callback: CallbackQuery) -> None:
    post_key = callback.data.split(":", 1)[1]
    pending_posts.pop(post_key, None)
    await callback.answer("Отменено.", show_alert=False)
    if callback.message:
        await callback.message.edit_reply_markup(reply_markup=None)
        await callback.message.reply("❌ Публикация отменена.")


# ---------------------------------------------------------------------------
# Handler: plain text (outside FSM) — treat as extra info hint
# ---------------------------------------------------------------------------


@router.message(F.text & ~F.text.startswith("/"), StateFilter(None))
async def handle_plain_text(message: Message) -> None:
    await message.answer(
        f"Чтобы создать пост, отправьте мне фото автомобиля.\n"
        f"Вы можете добавить описание в подписи к фото."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    config = BotConfig.load()

    if not config.is_configured():
        logger.warning("Bot is not configured. Starting in setup mode...")
        temp_token = input("Введи временный Telegram токен для первой настройки: ").strip()
        bot = Bot(token=temp_token)
        storage = MemoryStorage()
        dp = Dispatcher(storage=storage)

        dp.message.register(cmd_setup, F.text == "/setup")
        dp.message.register(cmd_start, F.text == "/start")
        dp.message.register(setup_telegram_token, ConfigStates.waiting_telegram_token)
        dp.message.register(setup_openai_key, ConfigStates.waiting_openai_key)
        dp.message.register(setup_channel_id, ConfigStates.waiting_channel_id)

        logger.info("🔧 Setup mode started. Use /setup command")
        try:
            await dp.start_polling(bot, allowed_updates=["message"])
        finally:
            await bot.session.close()
        return

    bot = Bot(token=config.telegram_token)
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(router)

    logger.info("Starting Zera Auto Bot...")
    try:
        await dp.start_polling(bot, allowed_updates=["message", "callback_query"])
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
