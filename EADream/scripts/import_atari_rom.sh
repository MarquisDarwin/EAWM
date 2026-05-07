wget https://www.atarimania.com/roms/Atari-2600-VCS-ROM-Collection.zip
unzip Atari-2600-VCS-ROM-Collection.zip -d AtariROMS
python -m atari_py.import_roms AtariROMS
rm -rf AtariROMS