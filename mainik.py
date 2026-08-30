from embedding import encode
from palette_generator import PaletteGenerator


def main():

    generator = PaletteGenerator(
        model_path="palette_network.pth"
    )

    print("Palette generator is ready.")

    while True:

        query = input(
            "\nDescribe a palette (or 'exit'): "
        )

        if query.lower() == "exit":
            break

        if not query.strip():
            print("Please enter a description.")
            continue

        # Convert the user's text into a 384-dimensional embedding
        embedding = encode([query])[0]

        # Generate a new palette using the neural network
        generated = generator.generate(
            embedding
        )

        print("\nGenerated palette:")

        print("\nHEX:")

        for color in generated["hex"]:
            print(color)

        print("\nRGB:")

        for color in generated["rgb"]:
            print(color)

        print("\nLAB:")

        for color in generated["lab"]:
            print(
                tuple(
                    round(value, 2)
                    for value in color
                )
            )


if __name__ == "__main__":
    main()