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
from dataclasses import dataclass
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
EMOJI_MONEY = "💰"
EMOJI_PHONE = "📱"
EMOJI_CHECK = "✅"
EMOJI_WRENCH = "🔧"
EMOJI_STAR = "⭐"
EMOJI_ROCKET = "🚀"

# ---------------------------------------------------------------------------
# Constants — non-bold label words for caption entities
# ---------------------------------------------------------------------------

NON_BOLD_WORDS_ICE = {"двигатель", "пробег", "комплектация", "состояние"}
NON_BOLD_WORDS_EV = {"электромотор", "батарея", "запас", "хода", "пробег", "кузов", "комплектация", "состояние", "заявленный"}

# ---------------------------------------------------------------------------
# Constants — GPT prompt
# ---------------------------------------------------------------------------

GPT_SYSTEM_PROMPT = """
═══════════════════════════════════════════════════════════════════════════════
0. ОПРЕДЕЛЕНИЕ ТИПА АВТОМОБИЛЯ (КРИТИЧЕСКИ ВАЖНО!)
═══════════════════════════════════════════════════════════════════════════════

🔋 ПРАВИЛО #0: ОПРЕДЕЛИ ТИП АВТОМОБИЛЯ!

КЛЮЧЕВЫЕ СЛОВА для ЭЛЕКТРОМОБИЛЯ:
- "электромобиль", "электро", "⚡", "EV"
- "батарейка", "батарея", "kWh", "kw"
- "запас хода", "км хода"

ЕСЛИ ЭТО ЭЛЕКТРОМОБИЛЬ:
- Установи `is_electric: true`
- Заполни: motor_power_hp, battery_kwh, range_km, body_type
- НЕ заполняй поле `engine`

ЕСЛИ ЭТО ДВС:
- Установи `is_electric: false`
- Заполняй как обычно: engine, power_hp

═══════════════════════════════════════════════════════════════════════════════
1. ОСНОВНОЙ АНАЛИЗ
═══════════════════════════════════════════════════════════════════════════════

Ты — эксперт по продаже автомобилей. Анализируй фотографии и возвращай ТОЛЬКО валидный JSON.

Определи по фото:
- Марку, модель, год выпуска, комплектацию и состояние
- Для ДВС: тип двигателя (объём, тип коробки, топливо)
- Для электромобиля: мощность мотора, ёмкость батареи, запас хода, тип кузова
- Пробег (если виден на одометре), иначе оставь пустым
- Индексы фотографий (начиная с 0), на которых виден VIN-номер или гос. номер

Придумай короткое название автомобиля (марка + модель + год + ключевые данные).
Придумай рекламный слоган (1–2 предложения).
Оцени рыночную стоимость в рублях для Уссурийска и для Москвы.

Верни ТОЛЬКО валидный JSON без markdown-блоков:
{
  "is_electric": false,
  "title": "Toyota Camry 2020 2.5 AT",
  "slogan": "Рекламный слоган",
  "engine": "2.5 AT (бензин)",
  "mileage": "50 000 км",
  "trim": "Prestige",
  "condition": "Хорошее",
  "price_ussuriisk": 2500000,
  "price_moscow": 2700000,
  "power_hp": 181,
  "year": 2020,
  "flags": "не бита, не крашена",
  "vin_photo_indices": [],
  "notes": "",
  "motor_power_hp": null,
  "battery_kwh": null,
  "range_km": null,
  "body_type": ""
}

Для электромобилей: engine = "", power_hp = null; motor_power_hp, battery_kwh, range_km, body_type — обязательны.
Для ДВС: motor_power_hp = null, battery_kwh = null, range_km = null, body_type = "".
"""

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
    title: str = ""
    slogan: str = ""
    engine: str = ""
    mileage: str = ""
    trim: str = ""
    condition: str = ""
    price_ussuriisk: Optional[int] = None
    price_moscow: Optional[int] = None
    power_hp: Optional[int] = None
    year: Optional[int] = None
    flags: str = ""
    vin_photo_indices: List[int] = None
    notes: str = ""

    # NEW EV FIELDS ONLY:
    is_electric: bool = False
    motor_power_hp: Optional[int] = None
    battery_kwh: Optional[float] = None
    range_km: Optional[int] = None
    body_type: str = ""

    def __post_init__(self):
        if self.vin_photo_indices is None:
            self.vin_photo_indices = []


# ---------------------------------------------------------------------------
# Safe type helpers
# ---------------------------------------------------------------------------


def _safe_int(val: Any) -> Optional[int]:
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


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

    async def analyze_car(
        self,
        photos_bytes: List[bytes],
        extra_hint: str = "",
    ) -> Optional[CarInfo]:
        """Send photos to GPT-4o and return parsed CarInfo or None on failure."""
        if not photos_bytes:
            return None

        system_text = GPT_SYSTEM_PROMPT
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
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse GPT response: %s", exc)
            logger.debug("Raw GPT response: %s", data)
            return None

        is_electric = parsed.get("is_electric", False)

        car_info = CarInfo(
            title=parsed.get("title", ""),
            slogan=parsed.get("slogan", ""),
            engine=parsed.get("engine", ""),
            mileage=parsed.get("mileage", ""),
            trim=parsed.get("trim", ""),
            condition=parsed.get("condition", ""),
            price_ussuriisk=_safe_int(parsed.get("price_ussuriisk")),
            price_moscow=_safe_int(parsed.get("price_moscow")),
            power_hp=_safe_int(parsed.get("power_hp")),
            year=_safe_int(parsed.get("year")),
            flags=parsed.get("flags", ""),
            vin_photo_indices=list(parsed.get("vin_photo_indices") or []),
            notes=parsed.get("notes", ""),
            is_electric=is_electric,
        )

        if is_electric:
            car_info.motor_power_hp = _safe_int(parsed.get("motor_power_hp"))
            car_info.battery_kwh = _safe_float(parsed.get("battery_kwh"))
            car_info.range_km = _safe_int(parsed.get("range_km"))
            car_info.body_type = parsed.get("body_type", "")

        return car_info


# ---------------------------------------------------------------------------
# Caption builder with Telegram entities
# ---------------------------------------------------------------------------


def build_caption_with_entities(
    ci: CarInfo, manager_handle: str, rub_bonus: int
):
    """Build post caption text and list of Telegram MessageEntity dicts."""
    # Each segment is (text, is_bold)
    segments: List[tuple] = []

    # Title — bold
    title_prefix = f"{EMOJI_LIGHTNING} " if ci.is_electric else f"{EMOJI_CAR} "
    segments.append((title_prefix + ci.title + "\n", True))

    # Slogan
    if ci.slogan:
        segments.append(("\n" + ci.slogan + "\n", False))

    segments.append(("\n", False))

    # Characteristics block
    if ci.is_electric:
        # EV format
        if ci.motor_power_hp:
            segments.append(("— электромотор: ", False))
            segments.append((f"{ci.motor_power_hp} л.с.\n", True))
        if ci.battery_kwh:
            segments.append(("— батарея: ", False))
            segments.append((f"{ci.battery_kwh} kWh\n", True))
        if ci.range_km:
            segments.append(("— заявленный запас хода: ", False))
            segments.append((f"{ci.range_km} км\n", True))
        if ci.mileage:
            segments.append(("— пробег: ", False))
            segments.append((f"{ci.mileage}\n", True))
        if ci.body_type:
            segments.append(("— кузов: ", False))
            segments.append((f"{ci.body_type}\n", True))
        if ci.trim:
            segments.append(("— комплектация: ", False))
            segments.append((f"{ci.trim}\n", True))
        if ci.condition:
            segments.append(("— состояние: ", False))
            segments.append((f"{ci.condition}\n", True))
    else:
        # ICE format (KEEP EXACTLY AS IS)
        segments.append(("— двигатель: ", False))
        segments.append((f"{ci.engine}\n", True))
        if ci.mileage:
            segments.append(("— пробег: ", False))
            segments.append((f"{ci.mileage}\n", True))
        if ci.trim:
            segments.append(("— комплектация: ", False))
            segments.append((f"{ci.trim}\n", True))
        if ci.condition:
            segments.append(("— состояние: ", False))
            segments.append((f"{ci.condition}\n", True))

    # Flags / notes
    if ci.flags:
        segments.append(("\n" + ci.flags + "\n", False))

    # Prices
    segments.append(("\n", False))
    if ci.price_ussuriisk:
        total = ci.price_ussuriisk + rub_bonus
        price_str = f"{total:,}".replace(",", " ") + " ₽\n"
        segments.append((f"{EMOJI_MONEY} Уссурийск: ", False))
        segments.append((price_str, True))
    if ci.price_moscow:
        total = ci.price_moscow + rub_bonus
        price_str = f"{total:,}".replace(",", " ") + " ₽\n"
        segments.append((f"{EMOJI_MONEY} Москва: ", False))
        segments.append((price_str, True))

    # Contact
    if manager_handle:
        segments.append(("\n", False))
        segments.append((f"{EMOJI_PHONE} {manager_handle}\n", False))

    # Build full text and entity list
    full_text = ""
    entities: List[Dict[str, Any]] = []
    for seg_text, is_bold in segments:
        if not seg_text:
            continue
        if is_bold and seg_text.strip():
            offset = utf16_len(full_text)
            length = utf16_len(seg_text)
            if length > 0:
                entities.append(build_entity("bold", offset, length))
        full_text += seg_text

    return full_text, entities


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
    analyzer = ChatGPTAnalyzer(config.openai_api_key)
    car = await analyzer.analyze_car(photos_bytes, extra_hint=extra_text)

    if car is None:
        await status_msg.edit_text(
            "❌ Не удалось проанализировать фото. Попробуйте ещё раз или другое фото."
        )
        return

    # Filter out photos with VIN/plate numbers (don't block — just remove those photos)
    if car.vin_photo_indices:
        photo_ids = [
            fid for i, fid in enumerate(photo_ids)
            if i not in car.vin_photo_indices
        ]
        if not photo_ids:
            await status_msg.edit_text(
                "⚠️ Все фотографии содержат VIN-номер или гос. номер.\n"
                "Пожалуйста, добавьте фото без регистрационных данных."
            )
            return

    # Build caption with Telegram entities
    post_text, post_entities = build_caption_with_entities(
        car, config.manager_handle, config.rub_bonus
    )

    # Store pending post
    post_key = f"{chat_id}_{int(time.monotonic() * 1000)}"
    pending_posts[post_key] = {
        "text": post_text,
        "entities": post_entities,
        "photo_ids": photo_ids,
        "car": car,
    }

    # Send preview
    vehicle_label = f"{EMOJI_LIGHTNING} Электромобиль" if car.is_electric else f"{EMOJI_CAR} Автомобиль"
    preview_header = (
        f"{vehicle_label} распознан: {car.title}\n"
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
            caption_entities=post_entities,
            reply_markup=make_confirm_keyboard(post_key),
        )
    else:
        from aiogram.types import InputMediaPhoto

        media = [
            InputMediaPhoto(
                media=fid,
                caption=post_text if i == 0 else None,
                caption_entities=post_entities if i == 0 else None,
            )
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
    post_entities: List[Dict[str, Any]] = post.get("entities", [])

    try:
        if len(photo_ids) == 1:
            await bot.send_photo(
                chat_id=config.channel_id,
                photo=photo_ids[0],
                caption=post_text,
                caption_entities=post_entities,
            )
        else:
            from aiogram.types import InputMediaPhoto

            media = [
                InputMediaPhoto(
                    media=fid,
                    caption=post_text if i == 0 else None,
                    caption_entities=post_entities if i == 0 else None,
                )
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
