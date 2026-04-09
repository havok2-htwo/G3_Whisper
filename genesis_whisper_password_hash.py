import argparse
import getpass
import hashlib
import os


def build_pbkdf2_hash(password: str, iterations: int = 390000) -> str:
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def main():
    parser = argparse.ArgumentParser(description="Erzeugt einen PBKDF2-Hash fuer GENESIS_ADMIN_PASSWORD_HASH.")
    parser.add_argument("--password", help="Passwort im Klartext. Wenn leer, wird interaktiv gefragt.")
    parser.add_argument("--iterations", type=int, default=390000)
    args = parser.parse_args()

    password = args.password or getpass.getpass("Admin-Passwort: ")
    print(build_pbkdf2_hash(password, args.iterations))


if __name__ == "__main__":
    main()
