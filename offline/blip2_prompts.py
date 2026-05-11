"""
Category-specific BLIP-2 prompts for structured fashion captioning.
Keys match DeepFashion folder names exactly (extracted from crop path at runtime).

v3 changes:
- Switched from fill-in-the-blank [value] format to Q&A style
- States garment type as fact first ("This image shows a dress.")
- Asks direct questions about each attribute
- Uses "Answer as: X: Y" format — no brackets, no parenthetical options
- Explicit ignore instructions for bottom garments to avoid top description
"""

CATEGORY_PROMPTS: dict[str, str] = {
    "Dresses": (
        "This image shows a dress. "
        "Describe the dress with these details: "
        "What is the length? What is the neckline style? "
        "What are the sleeves like? What color is it? "
        "What pattern does it have? What fabric does it appear to be? "
        "Answer as: length: X, neckline: X, sleeves: X, color: X, pattern: X, fabric: X"
    ),
    "Blouses_Shirts": (
        "This image shows a blouse or shirt. "
        "Describe only the top garment with these details: "
        "What style is it? What is the neckline? "
        "What are the sleeves like? What color is it? "
        "What pattern does it have? How does it fit? "
        "Answer as: style: X, neckline: X, sleeves: X, color: X, pattern: X, fit: X"
    ),
    "Pants": (
        "This image shows a pair of pants. "
        "Describe only the pants with these details: "
        "What is the cut? What is the rise? "
        "What color are they? What fabric are they made of? "
        "How do they fit? "
        "Answer as: cut: X, rise: X, color: X, fabric: X, fit: X"
    ),
    "Denim": (
        "This image shows a denim garment. "
        "Describe only the denim item with these details: "
        "What is the cut or style? What is the rise? "
        "What color or wash is it? Is there any distressing? "
        "How does it fit? "
        "Answer as: cut: X, rise: X, color: X, distressing: X, fit: X"
    ),
    "Skirts": (
        "This image shows a skirt. "
        "Describe only the skirt with these details: "
        "What is the length? What is the silhouette? "
        "What color is it? What pattern does it have? "
        "What fabric does it appear to be? "
        "Answer as: length: X, silhouette: X, color: X, pattern: X, fabric: X"
    ),
    "Sweaters": (
        "This image shows a sweater. "
        "Describe only the sweater with these details: "
        "What style is it? What is the neckline? "
        "What are the sleeves like? What color is it? "
        "What pattern does it have? How heavy is the knit? "
        "Answer as: style: X, neckline: X, sleeves: X, color: X, pattern: X, weight: X"
    ),
    "Cardigans": (
        "This image shows a cardigan. "
        "Describe only the cardigan with these details: "
        "What style is it? How does it close? "
        "What are the sleeves like? What color is it? "
        "What pattern does it have? How does it fit? "
        "Answer as: style: X, closure: X, sleeves: X, color: X, pattern: X, fit: X"
    ),
    "Jackets_Coats": (
        "This image shows a jacket or coat. "
        "Describe only the outerwear with these details: "
        "What style is it? How does it close? "
        "What is the length? What color is it? "
        "What fabric does it appear to be? How does it fit? "
        "Answer as: style: X, closure: X, length: X, color: X, fabric: X, fit: X"
    ),
    "Jackets_Vests": (
        "This image shows a jacket or vest. "
        "Describe only the jacket or vest with these details: "
        "What style is it? How does it close? "
        "What is the length? What color is it? "
        "What fabric does it appear to be? How does it fit? "
        "Answer as: style: X, closure: X, length: X, color: X, fabric: X, fit: X"
    ),
    "Shorts": (
        "This image shows a pair of shorts. "
        "Describe only the shorts and ignore any top or shirt worn with them. "
        "What is the length? What is the rise? "
        "What color are they? What fabric are they made of? "
        "How do they fit? "
        "Answer as: length: X, rise: X, color: X, fabric: X, fit: X"
    ),
    "Tees_Tanks": (
        "This image shows a t-shirt or tank top. "
        "Describe only the top with these details: "
        "What style is it? What is the neckline? "
        "What are the sleeves like? What color is it? "
        "What pattern does it have? How does it fit? "
        "Answer as: style: X, neckline: X, sleeves: X, color: X, pattern: X, fit: X"
    ),
    "Graphic_Tees": (
        "This image shows a graphic t-shirt. "
        "Describe only the t-shirt with these details: "
        "What style is it? What graphic or text is printed on it? "
        "What is the base color of the shirt? How does it fit? "
        "Answer as: style: X, graphic: X, color: X, fit: X"
    ),
    "Rompers_Jumpsuits": (
        "This image shows a romper or jumpsuit. "
        "Describe only the garment with these details: "
        "Is it a romper or jumpsuit? What is the neckline? "
        "What are the sleeves like? What color is it? "
        "What pattern does it have? "
        "Answer as: silhouette: X, neckline: X, sleeves: X, color: X, pattern: X"
    ),
    "Leggings": (
        "This image shows a pair of leggings. "
        "Describe only the leggings and ignore any top or jacket worn with them. "
        "What is the length? What color are they? "
        "What pattern do they have? What fabric are they made of? "
        "What is the waistband like? "
        "Answer as: length: X, color: X, pattern: X, fabric: X, waistband: X"
    ),
    "Sweatshirts_Hoodies": (
        "This image shows a sweatshirt or hoodie. "
        "Describe only the sweatshirt with these details: "
        "Is it a hoodie or crewneck sweatshirt? Does it have a hood? "
        "What color is it? Is there any graphic or logo on it? "
        "How does it fit? "
        "Answer as: style: X, hood: X, color: X, graphic: X, fit: X"
    ),
    "Suiting": (
        "This image shows a suit or blazer. "
        "Describe only the suiting item with these details: "
        "What style is it? What is the lapel like? "
        "What color is it? What fabric does it appear to be? "
        "How does it fit? "
        "Answer as: style: X, lapel: X, color: X, fabric: X, fit: X"
    ),
    "Shirts_Polos": (
        "This image shows a shirt or polo. "
        "Describe only the shirt with these details: "
        "What style is it? What is the collar like? "
        "What are the sleeves like? What color is it? "
        "What pattern does it have? How does it fit? "
        "Answer as: style: X, collar: X, sleeves: X, color: X, pattern: X, fit: X"
    ),
}

DEFAULT_PROMPT: str = (
    "This image shows a clothing item. "
    "Describe only the clothing with these details: "
    "What type of garment is it? What color is it? "
    "What pattern does it have? What fabric does it appear to be? "
    "How does it fit? What are any notable style details? "
    "Answer as: garment: X, color: X, pattern: X, fabric: X, fit: X, details: X"
)


def get_prompt(category: str) -> str:
    """Return the structured Q&A prompt for a given DeepFashion category folder name."""
    return CATEGORY_PROMPTS.get(category, DEFAULT_PROMPT)