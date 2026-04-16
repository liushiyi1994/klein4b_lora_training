from PIL import Image, ImageDraw


def make_four_up_grid(images: list[Image.Image], labels: list[str]) -> Image.Image:
    if len(images) != 4:
        raise ValueError("make_four_up_grid requires exactly 4 images")
    if len(labels) != 4:
        raise ValueError("make_four_up_grid requires exactly 4 labels")

    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError("make_four_up_grid requires all images to have the same size")

    canvas = Image.new("RGB", (width * 4, height + 28), color="black")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        canvas.paste(image, (index * width, 28))
        draw.text((index * width + 4, 6), label, fill="white")
    return canvas
