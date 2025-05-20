# TSA Software Development - 2025

Pennsylvania TSA team 2013-901 submission for the software development event for nationals 2025.

## Run the program

Install npm and python

Arch-based:
```
# pacman -S python nodejs npm
```

Debian-based:
```
# sudo apt install python3 nodejs npm
```

Create virtual environment

```
$ python -m venv .venv
$ source .venv/bin/activate
```

Install dependencies:

```
$ npm install
$ pip install -r requirements.txt
```

Open two terminals and run the following commands simultaneously:

```
$ npm run dev
```

```
$ python -m new_backend.main
```

Allowed Extensions are png, jpg, jpeg, gif, heic, hevc

A new browser window should automatically open. If not, open to [http://localhost:5173](http://localhost:5173)
