from pathlib import Path

from matplotlib import pyplot as plt

from utils import BSSData, confirm, get_valid_int

NETWORKS_PATH = Path(__file__).parent.joinpath("networks")
IMAGES_PATH = Path(__file__).parent.joinpath("images")


def gen_images(networks_path: Path, images_path: Path, refresh: bool = False) -> None:
    networks = [network.name for network in networks_path.iterdir() if network.is_dir()]
    if not networks:
        print("No networks found.")
        return
    if refresh:
        if confirm("Are you sure you want to delete all existing images? (y/n)\n"):
            for image in images_path.iterdir():
                print("Removing all existing images ...")
                image.unlink()
    existing_images = [image.name[:-4] for image in images_path.iterdir() if image.is_file()]
    for network in networks:
        if network in existing_images:
            continue
        try:
            print(f"Generating image for {network} ...")
            bss_data = BSSData.from_files(networks_path.joinpath(network))
            bss_data.draw_graph_pretty(title=network, draw_dimensions=True)
            plt.savefig(images_path.joinpath(network + ".png"))
        except Exception as e:
            print(f"Error generating image for {network}: {e}")


def main() -> None:
    NETWORKS_PATH.mkdir(exist_ok=True)
    IMAGES_PATH.mkdir(exist_ok=True)
    while True:
        option = get_valid_int("What would you like to do?\n1) Generate new images\n2) Refresh all images\n3) Exit\n", 1, 3)
        if option == 1:
            gen_images(NETWORKS_PATH, IMAGES_PATH, False)
        elif option == 2:
            gen_images(NETWORKS_PATH, IMAGES_PATH, True)
        elif option == 3:
            break


if __name__ == "__main__":
    main()
