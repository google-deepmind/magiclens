## CIRCO
```
mkdir circo
cd circo
```

In folder `magic_lens/data/circo`

### Download annotations

```
git clone https://github.com/miccunifi/CIRCO.git
mv ./CIRCO/annotations ./
# rm -rf ./CIRCO (unused)
```

### Download images
```
mkdir COCO2017_unlabeled
# download images
wget http://images.cocodataset.org/zips/unlabeled2017.zip
unzip unlabeled2017.zip -d COCO2017_unlabeled
# download info
wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip
unzip image_info_unlabeled2017.zip -d COCO2017_unlabeled
```

The file structure should look like this (w/o showing CIRCO repo):
```
circo
└─── annotations
        | test.json
        | val.json
└─── COCO2017_unlabeled
    └─── annotations
        | image_info_unlabeled2017.json

    └─── unlabeled2017
        | 000000243611.jpg
        | 000000535009.jpg
        | 000000097553.jpg
        | ...
```


## FashionIQ (FIQ)
```
mkdir fiq
cd fiq
```

### Download FIQ
Donload dataset following [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq/issues/18)

Uncompress `images.tar.gz`, `image_splits-20220326T130551Z-001.zip`, and `captions-20220326T130604Z-001.zip` accordingly.

The file structure should look like this (w/o showing train/test.json):
```
├── README.md
├── captions
│   ├── cap.dress.val.json
│   ├── cap.shirt.val.json
│   └── cap.toptee.val.json
├── image_splits
│   ├── split.dress.val.json
│   ├── split.shirt.val.json
│   └── split.toptee.val.json
├── images
│   ├── B00HINZY58.png
│   ├── B00I0XXRJU.png
│   ├── B00J2UZLNU.png
│   └── B00J66MDMC.png
    ...
```

