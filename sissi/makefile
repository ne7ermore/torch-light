DATAPATH=$(CURDIR)/data

defaut: mkdir_data d-mozart d-chopin d-beeth d-burgm d-haydn d-brahms

mkdir_data:
	mkdir -p $(DATAPATH)

d-mozart:
	curl http://www.piano-midi.de/zip/mozart.zip -o $(DATAPATH)/mozart.zip
	unzip $(DATAPATH)/mozart.zip -d $(DATAPATH)/mozart
	rm $(DATAPATH)/mozart.zip

d-chopin:
	curl http://www.piano-midi.de/zip/chopin.zip -o $(DATAPATH)/chopin.zip
	unzip $(DATAPATH)/chopin.zip -d $(DATAPATH)/chopin
	rm $(DATAPATH)/chopin.zip

d-beeth:
	curl http://www.piano-midi.de/zip/beeth.zip -o $(DATAPATH)/beeth.zip
	unzip $(DATAPATH)/beeth.zip -d $(DATAPATH)/beeth
	rm $(DATAPATH)/beeth.zip

d-burgm:
	curl http://www.piano-midi.de/zip/burgm.zip -o $(DATAPATH)/burgm.zip
	unzip $(DATAPATH)/burgm.zip -d $(DATAPATH)/burgm
	rm $(DATAPATH)/burgm.zip

d-haydn:
	curl http://www.piano-midi.de/zip/haydn.zip -o $(DATAPATH)/haydn.zip
	unzip $(DATAPATH)/haydn.zip -d $(DATAPATH)/haydn
	rm $(DATAPATH)/haydn.zip

d-brahms:
	curl http://www.piano-midi.de/zip/brahms.zip -o $(DATAPATH)/brahms.zip
	unzip $(DATAPATH)/brahms.zip -d $(DATAPATH)/brahms
	rm $(DATAPATH)/brahms.zip
