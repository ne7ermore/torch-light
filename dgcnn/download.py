import requests

DATAPATH = "https://dataset-bj.cdn.bcebos.com/sked/train_data.json"
DEVPATH = "https://dataset-bj.cdn.bcebos.com/sked/dev_data.json"


def main():
    r = requests.get(DATAPATH)
    with open("data/train_data.json", "wb") as code:
        code.write(r.content)

    r = requests.get(DEVPATH)
    with open("data/dev_data.json", "wb") as code:
        code.write(r.content)


if __name__ == "__main__":
    main()
