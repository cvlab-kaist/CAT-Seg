import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import copy


stuff_colors = [[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
               [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
               [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
               [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
               [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
               [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
               [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
               [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
               [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
               [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
               [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
               [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
               [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
               [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
               [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
               [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
               [0, 0, 230], [119, 11, 32],
               [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
               [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
               [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
               [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
               [64, 192, 96], [64, 160, 64], [64, 64, 0]]

def _get_pascal_context_59_meta():
    context_classes = ["aeroplane", "bag", "bed", "bedclothes", "bench", "bicycle", "bird", "boat", "book", "bottle", "building", "bus", "cabinet", "car", "cat", "ceiling", "chair", "cloth", "computer", "cow", "cup", "curtain", "dog", "door", "fence", "floor", "flower", "food", "grass", "ground", "horse", "keyboard", "light", "motorbike", "mountain", "mouse", "person", "plate", "platform", "pottedplant", "road", "rock", "sheep", "shelves", "sidewalk", "sign", "sky", "snow", "sofa", "diningtable", "track", "train", "tree", "truck", "tvmonitor", "wall", "water", "window", "wood"]
    context_colors = [stuff_colors[i % len(stuff_colors)] for i in range(len(context_classes))]
    ret = {
        "stuff_colors" : context_colors,
        "stuff_classes" : context_classes,
    }
    return ret

def register_pascal_context_59(root):
    root = os.path.join(root, "VOCdevkit", "VOC2010")
    meta = _get_pascal_context_59_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("test", "JPEGImages", "annotations_detectron2/pc59_val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"context_59_{name}_sem_seg"
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=255, **meta,)

def _get_pascal_context_459_meta():
    context_459_classes = ["accordion", "aeroplane", "airconditioner", "antenna", "artillery", "ashtray", "atrium", "babycarriage", "bag", "ball", "balloon", "bambooweaving", "barrel", "baseballbat", "basket", "basketballbackboard", "bathtub", "bed", "bedclothes", "beer", "bell", "bench", "bicycle", "binoculars", "bird", "birdcage", "birdfeeder", "birdnest", "blackboard", "board", "boat", "bone", "book", "bottle", "bottleopener", "bowl", "box", "bracelet", "brick", "bridge", "broom", "brush", "bucket", "building", "bus", "cabinet", "cabinetdoor", "cage", "cake", "calculator", "calendar", "camel", "camera", "cameralens", "can", "candle", "candleholder", "cap", "car", "card", "cart", "case", "casetterecorder", "cashregister", "cat", "cd", "cdplayer", "ceiling", "cellphone", "cello", "chain", "chair", "chessboard", "chicken", "chopstick", "clip", "clippers", "clock", "closet", "cloth", "clothestree", "coffee", "coffeemachine", "comb", "computer", "concrete", "cone", "container", "controlbooth", "controller", "cooker", "copyingmachine", "coral", "cork", "corkscrew", "counter", "court", "cow", "crabstick", "crane", "crate", "cross", "crutch", "cup", "curtain", "cushion", "cuttingboard", "dais", "disc", "disccase", "dishwasher", "dock", "dog", "dolphin", "door", "drainer", "dray", "drinkdispenser", "drinkingmachine", "drop", "drug", "drum", "drumkit", "duck", "dumbbell", "earphone", "earrings", "egg", "electricfan", "electriciron", "electricpot", "electricsaw", "electronickeyboard", "engine", "envelope", "equipment", "escalator", "exhibitionbooth", "extinguisher", "eyeglass", "fan", "faucet", "faxmachine", "fence", "ferriswheel", "fireextinguisher", "firehydrant", "fireplace", "fish", "fishtank", "fishbowl", "fishingnet", "fishingpole", "flag", "flagstaff", "flame", "flashlight", "floor", "flower", "fly", "foam", "food", "footbridge", "forceps", "fork", "forklift", "fountain", "fox", "frame", "fridge", "frog", "fruit", "funnel", "furnace", "gamecontroller", "gamemachine", "gascylinder", "gashood", "gasstove", "giftbox", "glass", "glassmarble", "globe", "glove", "goal", "grandstand", "grass", "gravestone", "ground", "guardrail", "guitar", "gun", "hammer", "handcart", "handle", "handrail", "hanger", "harddiskdrive", "hat", "hay", "headphone", "heater", "helicopter", "helmet", "holder", "hook", "horse", "horse-drawncarriage", "hot-airballoon", "hydrovalve", "ice", "inflatorpump", "ipod", "iron", "ironingboard", "jar", "kart", "kettle", "key", "keyboard", "kitchenrange", "kite", "knife", "knifeblock", "ladder", "laddertruck", "ladle", "laptop", "leaves", "lid", "lifebuoy", "light", "lightbulb", "lighter", "line", "lion", "lobster", "lock", "machine", "mailbox", "mannequin", "map", "mask", "mat", "matchbook", "mattress", "menu", "metal", "meterbox", "microphone", "microwave", "mirror", "missile", "model", "money", "monkey", "mop", "motorbike", "mountain", "mouse", "mousepad", "musicalinstrument", "napkin", "net", "newspaper", "oar", "ornament", "outlet", "oven", "oxygenbottle", "pack", "pan", "paper", "paperbox", "papercutter", "parachute", "parasol", "parterre", "patio", "pelage", "pen", "pencontainer", "pencil", "person", "photo", "piano", "picture", "pig", "pillar", "pillow", "pipe", "pitcher", "plant", "plastic", "plate", "platform", "player", "playground", "pliers", "plume", "poker", "pokerchip", "pole", "pooltable", "postcard", "poster", "pot", "pottedplant", "printer", "projector", "pumpkin", "rabbit", "racket", "radiator", "radio", "rail", "rake", "ramp", "rangehood", "receiver", "recorder", "recreationalmachines", "remotecontrol", "road", "robot", "rock", "rocket", "rockinghorse", "rope", "rug", "ruler", "runway", "saddle", "sand", "saw", "scale", "scanner", "scissors", "scoop", "screen", "screwdriver", "sculpture", "scythe", "sewer", "sewingmachine", "shed", "sheep", "shell", "shelves", "shoe", "shoppingcart", "shovel", "sidecar", "sidewalk", "sign", "signallight", "sink", "skateboard", "ski", "sky", "sled", "slippers", "smoke", "snail", "snake", "snow", "snowmobiles", "sofa", "spanner", "spatula", "speaker", "speedbump", "spicecontainer", "spoon", "sprayer", "squirrel", "stage", "stair", "stapler", "stick", "stickynote", "stone", "stool", "stove", "straw", "stretcher", "sun", "sunglass", "sunshade", "surveillancecamera", "swan", "sweeper", "swimring", "swimmingpool", "swing", "switch", "table", "tableware", "tank", "tap", "tape", "tarp", "telephone", "telephonebooth", "tent", "tire", "toaster", "toilet", "tong", "tool", "toothbrush", "towel", "toy", "toycar", "track", "train", "trampoline", "trashbin", "tray", "tree", "tricycle", "tripod", "trophy", "truck", "tube", "turtle", "tvmonitor", "tweezers", "typewriter", "umbrella", "unknown", "vacuumcleaner", "vendingmachine", "videocamera", "videogameconsole", "videoplayer", "videotape", "violin", "wakeboard", "wall", "wallet", "wardrobe", "washingmachine", "watch", "water", "waterdispenser", "waterpipe", "waterskateboard", "watermelon", "whale", "wharf", "wheel", "wheelchair", "window", "windowblinds", "wineglass", "wire", "wood", "wool"]
    context_colors = [stuff_colors[i % len(stuff_colors)] for i in range(len(context_459_classes))]
    ret = {
        "stuff_colors" : context_colors,
        "stuff_classes" : context_459_classes,
    }
    return ret

def register_pascal_context_459(root):
    root = os.path.join(root, "VOCdevkit", "VOC2010")
    meta = _get_pascal_context_459_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("test", "JPEGImages", "annotations_detectron2/pc459_val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"context_459_{name}_sem_seg"
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='tif', image_ext='jpg'))
        MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=459, **meta,)


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_pascal_context_59(_root)
register_pascal_context_459(_root)