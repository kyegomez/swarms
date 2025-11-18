/*
    * This file is used to delete all commands from the Discord API.
    * Only use this if you want to delete all commands and understand the consequences.
*/

require('dotenv').config();
const token = process.env.DISCORD_TOKEN;
const clientID = process.env.DISCORD_CLIENT_ID;
const guildID = process.env.DISCORD_GUILD_ID;
const { REST, Routes } = require('discord.js');
const fs = require('node:fs');

const rest = new REST({ version: '10' }).setToken(token);

rest.put(Routes.applicationCommands(clientID), { body: [] })
    .then(() => console.log('Successfully deleted application (/) commands.'))
    .catch(console.error);

rest.put(Routes.applicationGuildCommands(clientID, guildID), { body: [] })
    .then(() => console.log('Successfully deleted guild (/) commands.'))
    .catch(console.error);

