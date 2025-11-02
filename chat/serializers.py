# chat/serializers.py
from rest_framework import serializers
from .models import Chat, ChatMessage
from bots.serializers import BotSerializer

class ChatMessageSerializer(serializers.ModelSerializer):
    suggestions = serializers.SerializerMethodField()
    # --- NOVO: Campo para obter a URL do anexo ---
    attachment_url = serializers.SerializerMethodField()

    class Meta:
        model = ChatMessage
        # Adiciona os novos campos para LEITURA
        fields = ('id', 'chat', 'role', 'content', 'created_at', 'suggestions',
                  'attachment_url', 'attachment_type', 'original_filename')
        
        # --- CORREÇÃO 1: O PROBLEMA DO TEXTO SUMINDO ---
        # Antes: read_only_fields = fields (Isso tornava 'content' somente leitura)
        # Agora: Definimos explicitamente quais campos são 'read-only'.
        # 'content' e 'role' NÃO estão aqui, permitindo que a view de 
        # mensagem de texto (`ChatMessageListView`) os salve.
        read_only_fields = ('id', 'chat','role', 'created_at', 'suggestions',
                          'attachment_url', 'attachment_type', 'original_filename')

    def get_suggestions(self, obj):
        suggestions_list = []
        if obj.suggestion1:
            suggestions_list.append(obj.suggestion1)
        if obj.suggestion2:
            suggestions_list.append(obj.suggestion2)
        return suggestions_list

    # --- NOVO: Método para obter a URL completa ---
    def get_attachment_url(self, obj):
        if obj.attachment and hasattr(obj.attachment, 'url'):
            request = self.context.get('request', None) # Adiciona None como default
            if request:
                try:
                   # build_absolute_uri pode falhar em alguns contextos (ex: testes sem request)
                   return request.build_absolute_uri(obj.attachment.url)
                except Exception:
                    # Fallback para URL relativa se build_absolute_uri falhar
                    return obj.attachment.url
            return obj.attachment.url # Retorna URL relativa se não houver request
        return None

class ChatListSerializer(serializers.ModelSerializer):
    bot = BotSerializer(read_only=True)
    last_message = serializers.SerializerMethodField()

    class Meta:
        model = Chat
        fields = ('id', 'bot', 'last_message', 'last_message_at', 'status')

    def get_last_message(self, obj):
        last_msg = obj.messages.order_by('-created_at').first()
        if last_msg:
            # Passa o contexto (request) para o ChatMessageSerializer
            context = self.context
            return ChatMessageSerializer(last_msg, context=context).data
        return None

# --- NOVO: Serializer específico para Upload ---
class ChatMessageAttachmentSerializer(serializers.ModelSerializer):
    # Define 'attachment' como write-only para a criação/upload
    # Adiciona validação de tamanho aqui também (redundante com a view, mas bom ter)
    attachment = serializers.FileField(write_only=True, required=True, max_length=500) # max_length no FileField refere-se ao nome do ficheiro
    content = serializers.CharField(required=False, allow_blank=True, max_length=1000) # Limitar caption

    class Meta:
        model = ChatMessage
        fields = ('id', 'chat', 'role', 'content', 'created_at',
                  'attachment', 'attachment_type', 'original_filename')
        
        # --- CORREÇÃO 2: O PROBLEMA DO ANEXO SUMINDO ---
        # Antes: 'attachment_type' e 'original_filename' estavam aqui.
        # Agora: Removemos eles, permitindo que a view de anexo 
        # (`ChatMessageAttachmentView`) salve esses campos.
        read_only_fields = ('id', 'chat', 'role', 'created_at')

    def validate_attachment(self, value):
        # Validação existente...
        MAX_UPLOAD_SIZE = 10 * 1024 * 1024
        if value.size > MAX_UPLOAD_SIZE:
            raise serializers.ValidationError(f"File size cannot exceed {MAX_UPLOAD_SIZE // (1024*1024)}MB.")
        return value