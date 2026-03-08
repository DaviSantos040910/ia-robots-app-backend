import os
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from accounts.models import User

def reset_users():
    count = User.objects.count()
    print(f"--- Encontrados {count} usuários. Iniciando limpeza...")
    User.objects.all().delete()
    print("✅ Todos os usuários foram removidos com sucesso.")

if __name__ == "__main__":
    reset_users()
