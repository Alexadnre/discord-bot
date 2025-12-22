FROM node:20-bookworm

# Installation des outils de build et ffmpeg
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    libsodium-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie UNIQUEMENT des fichiers de dépendances d'abord
COPY package*.json ./

# Installation propre (incluant dotenv et les modules natifs)
RUN npm install

# Maintenant on copie le reste du code
COPY . .

ENV NODE_ENV=production

CMD ["node", "bot.js"]