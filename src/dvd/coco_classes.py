"""The 80 COCO detection categories as (category_id, name), in COCO order.

Lets the demo app and the mock agents run without downloading COCO annotations.
"""

# (category_id, name) pairs. COCO ids are NOT contiguous (1..90 with gaps), so the id is kept next to the name.
COCO_CATEGORIES = [
    (1, "person"), (2, "bicycle"), (3, "car"), (4, "motorcycle"), (5, "airplane"), (6, "bus"), (7, "train"),
    (8, "truck"), (9, "boat"), (10, "traffic light"), (11, "fire hydrant"), (13, "stop sign"),
    (14, "parking meter"), (15, "bench"), (16, "bird"), (17, "cat"), (18, "dog"), (19, "horse"), (20, "sheep"),
    (21, "cow"), (22, "elephant"), (23, "bear"), (24, "zebra"), (25, "giraffe"), (27, "backpack"),
    (28, "umbrella"), (31, "handbag"), (32, "tie"), (33, "suitcase"), (34, "frisbee"), (35, "skis"),
    (36, "snowboard"), (37, "sports ball"), (38, "kite"), (39, "baseball bat"), (40, "baseball glove"),
    (41, "skateboard"), (42, "surfboard"), (43, "tennis racket"), (44, "bottle"), (46, "wine glass"),
    (47, "cup"), (48, "fork"), (49, "knife"), (50, "spoon"), (51, "bowl"), (52, "banana"), (53, "apple"),
    (54, "sandwich"), (55, "orange"), (56, "broccoli"), (57, "carrot"), (58, "hot dog"), (59, "pizza"),
    (60, "donut"), (61, "cake"), (62, "chair"), (63, "couch"), (64, "potted plant"), (65, "bed"),
    (67, "dining table"), (70, "toilet"), (72, "tv"), (73, "laptop"), (74, "mouse"), (75, "remote"),
    (76, "keyboard"), (77, "cell phone"), (78, "microwave"), (79, "oven"), (80, "toaster"), (81, "sink"),
    (82, "refrigerator"), (84, "book"), (85, "clock"), (86, "vase"), (87, "scissors"), (88, "teddy bear"),
    (89, "hair drier"), (90, "toothbrush"),
]
# Just the class names, in COCO order (used as the detector vocabulary).
COCO_NAMES = [n for _, n in COCO_CATEGORIES]
# name -> COCO category id lookup.
COCO_NAME_TO_ID = {n: i for i, n in COCO_CATEGORIES}
# Sanity check: COCO has exactly 80 detection classes.
assert len(COCO_CATEGORIES) == 80
