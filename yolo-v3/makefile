defaut:
	wget -c http://images.cocodataset.org/zips/train2014.zip
	wget -c http://images.cocodataset.org/zips/val2014.zip
	wget -c https://pjreddie.com/media/files/coco/labels.tgz

	mv train2014.zip ./data/
	mv val2014.zip ./data/
	mv labels.tgz ./data/

	cd data && unzip -q train2014.zip
	cd data && unzip -q val2014.zip
	cd data && tar xzf labels.tgz
