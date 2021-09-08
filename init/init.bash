# install clear command
cp sources.list /etc/apt
apt update && apt upgrade
apt install --reinstall ncurses-bin

# install python package
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install -r requirements.txt


