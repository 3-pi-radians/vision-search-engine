"""
Category-specific BLIP-2 prompts for structured fashion captioning.
Keys match DeepFashion folder names exactly.

v4: Switched to single concise sentence format.
No templates, no placeholders, no structured format.
Model describes what it sees naturally.
"""

CATEGORY_PROMPTS: dict[str, str] = {
    "Dresses": (
        "This image shows a dress. "
        "Describe only the dress in one concise sentence using visible fashion attributes "
        "such as color, fit, sleeves, neckline, length, and style. "
        "If some attributes are unclear, omit them."
    ),
    "Blouses_Shirts": (
        "This image shows a blouse or shirt. "
        "Describe only the top garment in one concise sentence using visible fashion attributes "
        "such as color, fit, sleeves, neckline, pattern, and style. "
        "If some attributes are unclear, omit them."
    ),
    "Pants": (
        "This image shows a pair of pants. "
        "Describe only the pants in one concise sentence using visible fashion attributes "
        "such as color, cut, rise, fabric, and fit. "
        "If some attributes are unclear, omit them."
    ),
    "Denim": (
        "This image shows a denim garment. "
        "Describe only the denim item in one concise sentence using visible fashion attributes "
        "such as color, wash, cut, rise, distressing, and fit. "
        "If some attributes are unclear, omit them."
    ),
    "Skirts": (
        "This image shows a skirt. "
        "Describe only the skirt in one concise sentence using visible fashion attributes "
        "such as color, length, silhouette, pattern, and fabric. "
        "If some attributes are unclear, omit them."
    ),
    "Sweaters": (
        "This image shows a sweater. "
        "Describe only the sweater in one concise sentence using visible fashion attributes "
        "such as color, neckline, sleeves, pattern, knit weight, and style. "
        "If some attributes are unclear, omit them."
    ),
    "Cardigans": (
        "This image shows a cardigan. "
        "Describe only the cardigan in one concise sentence using visible fashion attributes "
        "such as color, closure, sleeves, pattern, fit, and style. "
        "If some attributes are unclear, omit them."
    ),
    "Jackets_Coats": (
        "This image shows a jacket or coat. "
        "Describe only the outerwear in one concise sentence using visible fashion attributes "
        "such as color, style, closure, length, fabric, and fit. "
        "If some attributes are unclear, omit them."
    ),
    "Jackets_Vests": (
        "This image shows a jacket or vest. "
        "Describe only the jacket or vest in one concise sentence using visible fashion attributes "
        "such as color, style, closure, length, fabric, and fit. "
        "If some attributes are unclear, omit them."
    ),
    "Shorts": (
        "This image shows a pair of shorts. "
        "Describe only the shorts in one concise sentence using visible fashion attributes "
        "such as color, length, fabric, rise, and fit. "
        "Do not describe any top or shirt worn with the shorts. "
        "If some attributes are unclear, omit them."
    ),
    "Tees_Tanks": (
        "This image shows a t-shirt or tank top. "
        "Describe only the top in one concise sentence using visible fashion attributes "
        "such as color, neckline, sleeves, pattern, fit, and style. "
        "If some attributes are unclear, omit them."
    ),
    "Graphic_Tees": (
        "This image shows a graphic t-shirt. "
        "Describe only the t-shirt in one concise sentence including the graphic or text on it "
        "and visible fashion attributes such as color, fit, and style. "
        "If some attributes are unclear, omit them."
    ),
    "Rompers_Jumpsuits": (
        "This image shows a romper or jumpsuit. "
        "Describe only the garment in one concise sentence using visible fashion attributes "
        "such as color, neckline, sleeves, pattern, length, and silhouette. "
        "If some attributes are unclear, omit them."
    ),
    "Leggings": (
        "This image shows a pair of leggings. "
        "Describe only the leggings in one concise sentence using visible fashion attributes "
        "such as color, length, pattern, fabric, and waistband. "
        "Do not describe any top or jacket worn with the leggings. "
        "If some attributes are unclear, omit them."
    ),
    "Sweatshirts_Hoodies": (
        "This image shows a sweatshirt or hoodie. "
        "Describe only the sweatshirt in one concise sentence using visible fashion attributes "
        "such as color, style, hood, graphic or logo, and fit. "
        "If some attributes are unclear, omit them."
    ),
    "Suiting": (
        "This image shows a suit or blazer. "
        "Describe only the suiting item in one concise sentence using visible fashion attributes "
        "such as color, style, lapel, fabric, and fit. "
        "If some attributes are unclear, omit them."
    ),
    "Shirts_Polos": (
        "This image shows a shirt or polo. "
        "Describe only the shirt in one concise sentence using visible fashion attributes "
        "such as color, style, collar, sleeves, pattern, and fit. "
        "If some attributes are unclear, omit them."
    ),
}

DEFAULT_PROMPT: str = (
    "This image shows a clothing item. "
    "Describe only the clothing in one concise sentence using visible fashion attributes "
    "such as color, style, fit, pattern, fabric, and notable details. "
    "If some attributes are unclear, omit them."
)


def get_prompt(category: str) -> str:
    """Return the sentence-style prompt for a given DeepFashion category."""
    return CATEGORY_PROMPTS.get(category, DEFAULT_PROMPT)