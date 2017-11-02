import json
import os


def getConfig():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(ROOT_DIR, 'config.conf')


    with open(CONFIG_PATH) as rf:
        content = rf.read()

    content = content.replace("\r\n", " ")
    config = json.loads(content)

    return config
