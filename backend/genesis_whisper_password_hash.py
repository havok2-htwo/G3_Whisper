import argparse
import getpass
import hashlib
import os
import secrets


def build_pbkdf2_hash(password: str, iterations: int = 390000) -> str:
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def verify_pbkdf2_hash(password: str, encoded: str) -> bool:
    """Constant-time verify of a password against a pbkdf2_sha256$iter$salt$digest string."""
    try:
        algorithm, iterations, salt, digest = (encoded or "").split("$", 3)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    try:
        expected = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iterations)
        ).hex()
    except (ValueError, TypeError):
        return False
    return secrets.compare_digest(expected, digest)


def main():
    parser = argparse.ArgumentParser(description="Legacy helper: erzeugt einen generischen PBKDF2-SHA256-Hash.")
    parser.add_argument("--password", help="Klartextwert. Wenn leer, wird interaktiv gefragt.")
    parser.add_argument("--iterations", type=int, default=390000)
    args = parser.parse_args()

    password = args.password or getpass.getpass("Wert zum Hashen: ")
    print(build_pbkdf2_hash(password, args.iterations))


if __name__ == "__main__":
    main()
