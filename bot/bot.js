require('dotenv').config();
const { Client, GatewayIntentBits, Events, REST, Routes } = require('discord.js');
const { joinVoiceChannel, EndBehaviorType } = require('@discordjs/voice');
const prism = require('prism-media');

// Augmenter la limite pour éviter les avertissements de fuite mémoire (Node.js)
require('events').EventEmitter.defaultMaxListeners = 50;

const STT_URL = process.env.STT_URL || 'http://stt:3000/transcribe';
const SILENCE_MS = parseInt(process.env.RECORD_SILENCE_MS || "2000");

const client = new Client({ 
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates] 
});

// Set pour éviter de traiter plusieurs flux simultanés pour le même utilisateur
const activeListeners = new Set();

client.once(Events.ClientReady, async () => {
    console.log(`🚀 Bot BobbY en ligne`);
    const rest = new REST({ version: '10' }).setToken(process.env.DISCORD_TOKEN);
    try {
        await rest.put(
            Routes.applicationGuildCommands(process.env.CLIENT_ID, process.env.GUILD_ID),
            { body: [{ name: 'join', description: 'Rejoint le vocal' }] }
        );
    } catch (e) { 
        console.error("❌ Erreur enregistrement commandes Slash:", e); 
    }
});

client.on(Events.InteractionCreate, async (interaction) => {
    if (!interaction.isChatInputCommand() || interaction.commandName !== 'join') return;

    const channel = interaction.member.voice.channel;
    if (!channel) return interaction.reply("Connecte-toi à un salon !");

    const connection = joinVoiceChannel({
        channelId: channel.id,
        guildId: channel.guild.id,
        adapterCreator: channel.guild.voiceAdapterCreator,
        selfDeaf: false,
    });

    await interaction.reply(`✅ BobbY est à l'écoute (Silence: ${SILENCE_MS}ms)`);

    connection.receiver.speaking.on('start', (userId) => {
        // Anti-doublon : Si on traite déjà cet utilisateur, on ignore les nouveaux paquets
        if (activeListeners.has(userId)) return;
        
        activeListeners.add(userId);

        // Capture de l'audio jusqu'au silence
        const audioStream = connection.receiver.subscribe(userId, {
            end: { behavior: EndBehaviorType.AfterSilence, duration: SILENCE_MS },
        });

        // Décodage Opus vers PCM Raw (format attendu par ton stt.py)
        const decoder = new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 });
        const streamForSTT = audioStream.pipe(decoder);

        // Envoi au service Python (STT + LLM intégré)
        fetch(STT_URL, {
            method: 'POST',
            headers: { 
                'Content-Type': 'audio/raw;rate=48000;bits=16;channels=2', 
                'X-User-Id': userId 
            },
            body: streamForSTT,
            duplex: 'half'
        })
        .then(res => res.json())
        .then((data) => {
            // "data.text" contient la réponse finale de l'IA renvoyée par ton service Python
            if (data && data.detected && data.text && data.text.length > 0) {
                console.log(`🤖 Réponse pour ${userId}: ${data.text}`);
                
                // Envoi direct dans le salon textuel
                interaction.channel.send(`**<@${userId}>** : ${data.text}`);
            }
        })
        .catch(err => {
            console.error(`❌ Erreur service STT:`, err.message);
        })
        .finally(() => {
            // NETTOYAGE CRUCIAL pour éviter les fuites mémoire et les crashs de flux
            activeListeners.delete(userId);
            decoder.destroy();
            audioStream.destroy();
        });
    });
});

client.login(process.env.DISCORD_TOKEN);