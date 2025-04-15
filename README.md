# TSA Software Development - 2025

Pennsylvania TSA team 2013-901 submission for the software development event for states 2025.

## Run the program

Install npm and python

Arch-based:
```
# pacman -S python nodejs npm
```

Debian-based:
```
# apt install python nodejs npm
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
$ npm start
```

```
$ python -m new_backend.main
```

Allowed Extensions are png, jpg, jpeg, and gif

A new browser window should automatically open. If not, open to [http://localhost:3000](http://localhost:3000)
