"""
Category-specific BLIP-2 prompts for structured fashion captioning.
Keys match DeepFashion folder names exactly (extracted from crop path at runtime).

Fix v2: Removed parenthetical options that caused model to copy template text.
Uses [value] placeholders and explicit "describe what you see" instruction.
"""

CATEGORY_PROMPTS: dict[str, str] = {
    "Dresses": (
        "Look only at the dress in this image. Ignore the person wearing it. "
        "Describe the dress by completing each attribute with what you actually see: "
        "length: [mini/midi/maxi/knee-length], "
        "neckline: [v-neck/crew/square/halter/off-shoulder/sweetheart], "
        "sleeves: [sleeveless/short/long/3-quarter], "
        "color: [color], "
        "pattern: [solid/floral/striped/printed/geometric], "
        "fabric: [fabric type]."
    ),
    "Blouses_Shirts": (
        "Look only at the top/blouse/shirt in this image. Ignore the person wearing it. "
        "Describe the garment by completing each attribute with what you actually see: "
        "style: [blouse/shirt/peasant/wrap/tunic], "
        "neckline: [v-neck/crew/collared/button-down/tie-neck], "
        "sleeves: [sleeveless/short/long/3-quarter/cap], "
        "color: [color], "
        "pattern: [solid/floral/striped/printed/plaid], "
        "fit: [relaxed/fitted/oversized/cropped]."
    ),
    "Pants": (
        "Look only at the pants in this image. Ignore the person wearing them. "
        "Describe the pants by completing each attribute with what you actually see: "
        "cut: [straight/wide-leg/tapered/flared/slim], "
        "rise: [high/mid/low], "
        "color: [color], "
        "fabric: [cotton/linen/satin/twill/knit], "
        "fit: [relaxed/fitted/skinny/loose]."
    ),
    "Denim": (
        "Look only at the denim item in this image. Ignore the person wearing it. "
        "Describe the denim by completing each attribute with what you actually see: "
        "cut: [straight/skinny/wide-leg/flared/tapered/bootcut], "
        "rise: [high/mid/low], "
        "color: [light-wash/medium-wash/dark-wash/black/white], "
        "distressing: [none/light/heavy/ripped], "
        "fit: [relaxed/fitted/skinny/loose]."
    ),
    "Skirts": (
        "Look only at the skirt in this image. Ignore the person wearing it. "
        "Describe the skirt by completing each attribute with what you actually see: "
        "length: [mini/midi/maxi/knee-length], "
        "silhouette: [a-line/pencil/wrap/pleated/flared/asymmetric], "
        "color: [color], "
        "pattern: [solid/floral/striped/printed/plaid], "
        "fabric: [cotton/satin/denim/lace/chiffon/knit]."
    ),
    "Sweaters": (
        "Look only at the sweater in this image. Ignore the person wearing it. "
        "Describe the sweater by completing each attribute with what you actually see: "
        "style: [pullover/turtleneck/crewneck/v-neck], "
        "sleeves: [long/short/sleeveless/3-quarter], "
        "color: [color], "
        "pattern: [solid/striped/cable-knit/fair-isle/ribbed], "
        "weight: [lightweight/medium/chunky]."
    ),
    "Cardigans": (
        "Look only at the cardigan in this image. Ignore the person wearing it. "
        "Describe the cardigan by completing each attribute with what you actually see: "
        "style: [open-front/button-down/longline/cropped/oversized], "
        "closure: [open/buttons/zipper], "
        "sleeves: [long/short/3-quarter], "
        "color: [color], "
        "pattern: [solid/striped/cable-knit/ribbed/printed], "
        "fit: [relaxed/fitted/oversized/cropped]."
    ),
    "Jackets_Coats": (
        "Look only at the jacket or coat in this image. Ignore the person wearing it. "
        "Describe the outerwear by completing each attribute with what you actually see: "
        "style: [blazer/trench/parka/bomber/moto/peacoat/duster], "
        "closure: [buttons/zipper/snap/open], "
        "length: [cropped/hip/knee/midi/maxi], "
        "color: [color], "
        "fabric: [denim/leather/wool/cotton/nylon/tweed], "
        "fit: [relaxed/fitted/oversized/structured]."
    ),
    "Jackets_Vests": (
        "Look only at the jacket or vest in this image. Ignore the person wearing it. "
        "Describe the garment by completing each attribute with what you actually see: "
        "style: [vest/puffer-vest/denim-jacket/bomber/utility], "
        "closure: [buttons/zipper/snap/open], "
        "length: [cropped/hip/long], "
        "color: [color], "
        "fabric: [denim/leather/nylon/cotton/quilted], "
        "fit: [relaxed/fitted/oversized/slim]."
    ),
    "Shorts": (
        "Look only at the shorts in this image. Ignore any top or jacket. "
        "Describe the shorts by completing each attribute with what you actually see: "
        "length: [mini/mid-thigh/knee-length/bermuda], "
        "rise: [high/mid/low], "
        "color: [color], "
        "fabric: [denim/cotton/linen/satin/knit/athletic], "
        "fit: [relaxed/fitted/wide-leg/tailored]."
    ),
    "Tees_Tanks": (
        "Look only at the tee or tank top in this image. Ignore the person wearing it. "
        "Describe the garment by completing each attribute with what you actually see: "
        "style: [t-shirt/tank/cami/muscle-tee/crop-top], "
        "neckline: [crew/v-neck/scoop/square/halter/racerback], "
        "sleeves: [sleeveless/short/long], "
        "color: [color], "
        "pattern: [solid/striped/printed/ribbed], "
        "fit: [relaxed/fitted/oversized/cropped]."
    ),
    "Graphic_Tees": (
        "Look only at the t-shirt in this image. Ignore the person wearing it. "
        "Describe the t-shirt by completing each attribute with what you actually see: "
        "style: [t-shirt/crop-top/oversized], "
        "graphic: [describe the graphic or text printed on it], "
        "color: [base color of the shirt], "
        "sleeves: [sleeveless/short], "
        "fit: [relaxed/fitted/oversized/cropped]."
    ),
    "Rompers_Jumpsuits": (
        "Look only at the romper or jumpsuit in this image. Ignore the person wearing it. "
        "Describe the garment by completing each attribute with what you actually see: "
        "silhouette: [romper/jumpsuit/playsuit/overalls], "
        "neckline: [v-neck/crew/halter/off-shoulder/square/strapless], "
        "sleeves: [sleeveless/short/long/3-quarter], "
        "color: [color], "
        "pattern: [solid/floral/striped/printed/geometric]."
    ),
    "Leggings": (
        "Look only at the leggings in this image. Ignore any top or jacket. "
        "Describe the leggings by completing each attribute with what you actually see: "
        "length: [full/capri/cropped/ankle], "
        "color: [color], "
        "pattern: [solid/striped/printed/color-block/camo], "
        "fabric: [cotton/spandex/velvet/mesh/leather-look], "
        "waistband: [elastic/wide-band/high-waist/fold-over]."
    ),
    "Sweatshirts_Hoodies": (
        "Look only at the sweatshirt or hoodie in this image. Ignore the person wearing it. "
        "Describe the garment by completing each attribute with what you actually see: "
        "style: [hoodie/crewneck-sweatshirt/zip-up/pullover], "
        "hood: [yes/no], "
        "color: [color], "
        "graphic: [none or describe any text/logo/pattern briefly], "
        "fit: [relaxed/fitted/oversized/cropped]."
    ),
    "Suiting": (
        "Look only at the suiting item in this image. Ignore the person wearing it. "
        "Describe the garment by completing each attribute with what you actually see: "
        "style: [blazer/suit-jacket/vest/trousers/suit], "
        "lapel: [notch/peak/shawl/collarless], "
        "color: [color], "
        "fabric: [wool/tweed/linen/cotton/velvet], "
        "fit: [slim/regular/relaxed/tailored/oversized]."
    ),
    "Shirts_Polos": (
        "Look only at the shirt or polo in this image. Ignore the person wearing it. "
        "Describe the garment by completing each attribute with what you actually see: "
        "style: [polo/button-down/henley/oxford/flannel], "
        "collar: [polo/spread/button-down/mandarin/banded], "
        "sleeves: [short/long/3-quarter], "
        "color: [color], "
        "pattern: [solid/striped/plaid/printed/checked], "
        "fit: [relaxed/fitted/slim/regular]."
    ),
}

DEFAULT_PROMPT: str = (
    "Look only at the clothing item in this image. Ignore the person wearing it. "
    "Describe the garment by completing each attribute with what you actually see: "
    "garment: [type of garment], "
    "color: [color], "
    "pattern: [solid/striped/printed/etc], "
    "fabric: [fabric type], "
    "fit: [relaxed/fitted/oversized/cropped/slim], "
    "details: [any notable style details]."
)


def get_prompt(category: str) -> str:
    """Return the prompt for a given DeepFashion category folder name."""
    return CATEGORY_PROMPTS.get(category, DEFAULT_PROMPT)