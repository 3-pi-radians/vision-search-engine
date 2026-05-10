"""
Category-specific BLIP-2 prompts for structured fashion captioning.
Keys match DeepFashion folder names exactly (extracted from crop path at runtime).
Each prompt: ignores the person, forces "attribute: value, ..." output format,
covers garment-type-specific attributes.
"""

CATEGORY_PROMPTS: dict[str, str] = {
    "Dresses": (
        "Ignore the person. Describe only the dress using this format: "
        "attribute: value, attribute: value. "
        "Attributes: length (mini/midi/maxi/knee-length), "
        "neckline (v-neck/crew/square/halter/off-shoulder/sweetheart), "
        "sleeves (sleeveless/short/long/3-quarter), "
        "color, pattern (solid/floral/striped/printed/geometric), fabric."
    ),
    "Blouses_Shirts": (
        "Ignore the person. Describe only the blouse or shirt using this format: "
        "attribute: value, attribute: value. "
        "Attributes: style (blouse/shirt/peasant/wrap/tunic), "
        "neckline (v-neck/crew/collared/button-down/tie-neck), "
        "sleeves (sleeveless/short/long/3-quarter/cap), "
        "color, pattern (solid/floral/striped/printed/plaid), fit (relaxed/fitted/oversized/cropped)."
    ),
    "Pants": (
        "Ignore the person. Describe only the pants using this format: "
        "attribute: value, attribute: value. "
        "Attributes: cut (straight/wide-leg/tapered/flared/slim), "
        "rise (high/mid/low), color, fabric (cotton/linen/satin/twill/knit), "
        "distressing (none/light/heavy), fit (relaxed/fitted/skinny/loose)."
    ),
    "Denim": (
        "Ignore the person. Describe only the denim item using this format: "
        "attribute: value, attribute: value. "
        "Attributes: cut (straight/skinny/wide-leg/flared/tapered/bootcut), "
        "rise (high/mid/low), color (light-wash/medium-wash/dark-wash/black/white), "
        "wash (clean/distressed/acid-wash/raw), "
        "distressing (none/light/heavy/ripped), fit (relaxed/fitted/skinny/loose)."
    ),
    "Skirts": (
        "Ignore the person. Describe only the skirt using this format: "
        "attribute: value, attribute: value. "
        "Attributes: length (mini/midi/maxi/knee-length), "
        "silhouette (a-line/pencil/wrap/pleated/flared/asymmetric), "
        "color, pattern (solid/floral/striped/printed/plaid), "
        "fabric (cotton/satin/denim/lace/chiffon/knit)."
    ),
    "Sweaters": (
        "Ignore the person. Describe only the sweater using this format: "
        "attribute: value, attribute: value. "
        "Attributes: style (pullover/turtleneck/mock-neck/crewneck/v-neck), "
        "neckline (crew/v-neck/turtleneck/mock-neck/cowl), "
        "sleeves (long/short/sleeveless/3-quarter), "
        "color, pattern (solid/striped/cable-knit/fair-isle/ribbed), "
        "weight (lightweight/medium/chunky)."
    ),
    "Cardigans": (
        "Ignore the person. Describe only the cardigan using this format: "
        "attribute: value, attribute: value. "
        "Attributes: style (open-front/button-down/longline/cropped/oversized), "
        "closure (open/buttons/zipper), "
        "sleeves (long/short/3-quarter), "
        "color, pattern (solid/striped/cable-knit/ribbed/printed), "
        "fit (relaxed/fitted/oversized/cropped)."
    ),
    "Jackets_Coats": (
        "Ignore the person. Describe only the jacket or coat using this format: "
        "attribute: value, attribute: value. "
        "Attributes: style (blazer/trench/parka/bomber/moto/peacoat/duster), "
        "closure (buttons/zipper/snap/open), "
        "length (cropped/hip/knee/midi/maxi), "
        "color, fabric (denim/leather/wool/cotton/nylon/tweed), "
        "fit (relaxed/fitted/oversized/structured)."
    ),
    "Jackets_Vests": (
        "Ignore the person. Describe only the jacket or vest using this format: "
        "attribute: value, attribute: value. "
        "Attributes: style (vest/gilet/puffer-vest/denim-jacket/bomber/utility), "
        "closure (buttons/zipper/snap/open), "
        "length (cropped/hip/long), "
        "color, fabric (denim/leather/nylon/cotton/quilted), "
        "fit (relaxed/fitted/oversized/slim)."
    ),
    "Shorts": (
        "Ignore the person. Describe only the shorts using this format: "
        "attribute: value, attribute: value. "
        "Attributes: length (mini/mid-thigh/knee-length/bermuda), "
        "rise (high/mid/low), "
        "color, fabric (denim/cotton/linen/satin/knit/athletic), "
        "fit (relaxed/fitted/wide-leg/tailored)."
    ),
    "Tees_Tanks": (
        "Ignore the person. Describe only the tee or tank top using this format: "
        "attribute: value, attribute: value. "
        "Attributes: style (t-shirt/tank/cami/muscle-tee/crop-top), "
        "neckline (crew/v-neck/scoop/square/halter/racerback), "
        "sleeves (sleeveless/short/long), "
        "color, pattern (solid/striped/printed/ribbed), "
        "fit (relaxed/fitted/oversized/cropped)."
    ),
    "Graphic_Tees": (
        "Ignore the person. Describe only the graphic tee using this format: "
        "attribute: value, attribute: value. "
        "Attributes: style (t-shirt/crop-top/oversized), "
        "graphic (describe the graphic or print briefly), "
        "color (base color of the shirt), "
        "fit (relaxed/fitted/oversized/cropped)."
    ),
    "Rompers_Jumpsuits": (
        "Ignore the person. Describe only the romper or jumpsuit using this format: "
        "attribute: value, attribute: value. "
        "Attributes: silhouette (romper/jumpsuit/playsuit/overalls), "
        "neckline (v-neck/crew/halter/off-shoulder/square/strapless), "
        "sleeves (sleeveless/short/long/3-quarter), "
        "color, pattern (solid/floral/striped/printed/geometric)."
    ),
    "Leggings": (
        "Ignore the person. Describe only the leggings using this format: "
        "attribute: value, attribute: value. "
        "Attributes: length (full/capri/cropped/ankle), "
        "color, pattern (solid/striped/printed/color-block/camo), "
        "fabric (cotton/spandex/velvet/mesh/leather-look), "
        "waistband (elastic/wide-band/high-waist/fold-over)."
    ),
    "Sweatshirts_Hoodies": (
        "Ignore the person. Describe only the sweatshirt or hoodie using this format: "
        "attribute: value, attribute: value. "
        "Attributes: style (hoodie/crewneck-sweatshirt/zip-up/pullover), "
        "hood (yes/no/drawstring), "
        "color, graphic (none/text/logo/pattern — describe briefly), "
        "fit (relaxed/fitted/oversized/cropped)."
    ),
    "Suiting": (
        "Ignore the person. Describe only the suiting item using this format: "
        "attribute: value, attribute: value. "
        "Attributes: style (blazer/suit-jacket/vest/trousers/suit), "
        "lapel (notch/peak/shawl/collarless), "
        "color, fabric (wool/tweed/linen/cotton/velvet/plaid), "
        "fit (slim/regular/relaxed/tailored/oversized)."
    ),
    "Shirts_Polos": (
        "Ignore the person. Describe only the shirt or polo using this format: "
        "attribute: value, attribute: value. "
        "Attributes: style (polo/button-down/henley/oxford/flannel), "
        "collar (polo/spread/button-down/mandarin/banded), "
        "sleeves (short/long/3-quarter), "
        "color, pattern (solid/striped/plaid/printed/checked), "
        "fit (relaxed/fitted/slim/regular)."
    ),
}

# Fallback for any category folder name not found in CATEGORY_PROMPTS
DEFAULT_PROMPT: str = (
    "Ignore the person. Describe only the clothing item using this format: "
    "attribute: value, attribute: value. "
    "Attributes: garment type, color, pattern (solid/striped/printed/etc), "
    "fabric, fit (relaxed/fitted/oversized/cropped/slim), style details."
)
