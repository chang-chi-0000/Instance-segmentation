# Instance-segmentation

## Reproduce
To reproduce you have to specify path to your datasets, and the structure of root should look like
```
root
├── dataset
│   ├── train_images
│   ├── pascal_train.json
│   ├── test.json
│   └── test_images
└── logs
```

### training
After including input images, just run `python hw3.py` then the program will run for 30 epochs
### testing
To produce the json file, run `python hw3.py --test <epoch>` and the json file`0650726.json` can be created
