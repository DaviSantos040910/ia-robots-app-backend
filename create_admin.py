import os
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from accounts.models import User

def create_admin():
    username = "admin"
    email = "davisantossousa2@gmail.com"
    password = "D@avi1914"

    if not User.objects.filter(username=username).exists():
        print(f"--- Criando superusuário: {username} ({email})...")
        User.objects.create_superuser(username=username, email=email, password=password)
        print(f"✅ Usuário '{username}' criado com sucesso!")
    else:
        print(f"⚠️ Usuário '{username}' já existe. Nenhuma ação necessária.")

if __name__ == "__main__":
    create_admin()
