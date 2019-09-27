# Get the stackoverflow data dump
sudo add-apt-repository -y ppa:transmissionbt/ppa
sudo apt-get -y update
sudo apt-get install -y transmission-cli transmission-common transmission-daemon

#sudo service transmission-daemon stop
# edit config

# files go in /var/lib/transmission-daemon/downloads/
sudo usermod -a -G debian-transmission $USER

sudo service transmission-daemon start

# Setup aliases for transmission download
alias t-start='sudo service transmission-daemon start'
alias t-stop='sudo service transmission-daemon stop'
alias t-reload='sudo service transmission-daemon reload'
alias t-list='transmission-remote -n 'transmission:transmission' -l'
alias t-basicstats='transmission-remote -n 'transmission:transmission' -st'
alias t-fullstats='transmission-remote -n 'transmission:transmission' -si'

# Get the torrent and start downloading
cd /var/lib/transmission/download
wget https://archive.org/download/stackexchange/stackexchange_archive.torrent
# transmission-remote -f -t /var/lib/transmission-daemon/downloads/stackexchange_archive.torrent
transmission-remote -n 'transmission:transmission' -a /var/lib/transmission-daemon/downloads/stackexchange_archive.torrent
# more torrent work needed, moving on...

# Need 7zip
# brew install p7zip
# sudo apt-get install -y p7zip

mkdir /nvm/stackexchange
cp /var/lib/transmission-daemon/downloads/stackexchange/* /nvm/stackexchange/
ln -s /nvm/stackexchange /home/rjurney/deep_products/data/

# Uncompress/recompress site data
mkdir stackoverflow
mv stackoverflow.com-* stackoverflow/
for f in *.7z;
do
    xml_file=(${f/stackoverflow.com-/})
    mv $f $xml_file
    7za x $xml_file
    lzof $xml_file
done

cd /home/rjurney/deep_products/data
mkdir GloVe
cd GloVe
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
