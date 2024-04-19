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
    existing_images = {image.stem: image for image in images_path.iterdir() if image.is_file()}
    for network in networks:
        image = existing_images.get(network)
        network_path = networks_path.joinpath(network)
        if image and image.stat().st_mtime > max(file.stat().st_mtime for file in network_path.glob('**/*') if file.is_file()):
            continue
        try:
            print(f"Generating image for {network} ...")
            bss_data = BSSData.from_files(network_path)
            bss_data.draw_graph_pretty(title=network, draw_dimensions=True)
            plt.savefig(images_path.joinpath(network + ".svg"))
            plt.clf()
        except Exception as e:
            print(f"Error generating image for {network}: {e}")


def gen_fixed_rings(networks_path: Path) -> None:
    networks = [network.name for network in networks_path.iterdir() if network.is_dir()]
    if not networks:
        print("No networks found.")
        return
    for network in networks:
        try:
            bss_data = BSSData.from_files(networks_path.joinpath(network))
            max_ring_node = max(bss_data.ring_network.nodes, key=lambda node: len(node.ring_neighbours))
            fixed_rings_path = networks_path.joinpath(network, "fixed_rings.txt")
            if not fixed_rings_path.exists():
                print(f"Creating fixed_rings.txt for {network} ...")
                with open(fixed_rings_path, "w+") as f:
                    f.write(f"{max_ring_node.id}\n")
                continue
            with open(fixed_rings_path, "r") as f:
                lines = f.readlines()
            if lines[0].strip() != str(max_ring_node.id):
                print(f"Updating fixed_rings.txt for {network} ...")
                with open(fixed_rings_path, "w") as f:
                    f.write(f"{max_ring_node.id}\n")
        except Exception as e:
            print(f"Error creating fixed_rings.txt for {network}: {e}")


def check_bond_lengths(networks_path: Path) -> None:
    for network in networks_path.iterdir():
        if network.is_dir():
            bss_data = BSSData.from_files(network)
            print(f"Network: {network.name:<30}\tBond length: {bss_data.base_network.get_average_bond_length()}")


def main() -> None:
    NETWORKS_PATH.mkdir(exist_ok=True)
    IMAGES_PATH.mkdir(exist_ok=True)
    while True:
        option = get_valid_int("What would you like to do?\n1) Generate new images\n"
                               "2) Refresh all images\n3) Auto-create fixed_rings.txt\n"
                               "4) Check bond lengths\n5) Exit\n", 1, 5)
        if option == 1:
            gen_images(NETWORKS_PATH, IMAGES_PATH, False)
        elif option == 2:
            gen_images(NETWORKS_PATH, IMAGES_PATH, True)
        elif option == 3:
            gen_fixed_rings(NETWORKS_PATH)
        elif option == 4:
            check_bond_lengths(NETWORKS_PATH)
        elif option == 5:
            break


if __name__ == "__main__":
    main()
