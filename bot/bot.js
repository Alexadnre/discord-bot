require('dotenv').config();
const { Client, GatewayIntentBits, Events, REST, Routes } = require('discord.js');
const { joinVoiceChannel, VoiceConnectionStatus, EndBehaviorType } = require('@discordjs/voice');
const prism = require('prism-media');

// Utilise l'URL du .env (http://stt:3000/transcribe)
const STT_URL = process.env.STT_URL; 
const SILENCE_MS = parseInt(process.env.RECORD_SILENCE_MS || "2000");

const client = new Client({ 
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates] 
});

client.once(Events.ClientReady, async () => {
    console.log(`🚀 Bot BobbY en ligne`);
    const rest = new REST({ version: '10' }).setToken(process.env.DISCORD_TOKEN);
    try {
        await rest.put(
            Routes.applicationGuildCommands(process.env.CLIENT_ID, process.env.GUILD_ID),
            { body: [{ name: 'join', description: 'Rejoint le vocal' }] }
        );
    } catch (e) { console.error("Erreur Slash:", e); }
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

    await interaction.reply(`Écoute active (Silence: ${SILENCE_MS}ms)`);

    connection.receiver.speaking.on('start', (userId) => {
        const audioStream = connection.receiver.subscribe(userId, {
            end: { behavior: EndBehaviorType.AfterSilence, duration: SILENCE_MS },
        });

        const decoder = new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 });

        fetch(STT_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'audio/raw;rate=48000;bits=16;channels=2', 'X-User-Id': userId },
            body: audioStream.pipe(decoder),
            duplex: 'half'
        })
        .then(res => res.json())
        .then(data => {
            if (data.text && data.text.length > 1) {
                interaction.channel.send(`**<@${userId}>** : ${data.text}`);
            }
        })
        .catch(err => console.error(`❌ Erreur STT:`, err.message));
    });
});

client.login(process.env.DISCORD_TOKEN);