/*
This script pushes all commands in the commands folder to be usable in discord.
*/

require('dotenv').config();
const token = process.env.DISCORD_TOKEN;
const clientID = process.env.DISCORD_CLIENT_ID;
const guildID = process.env.DISCORD_GUILD_ID;
const { REST, Routes } = require('discord.js');
const fs = require('node:fs');

const commands = [];

// Get all commands from the commands folder

const commandFiles = fs.readdirSync('./commands').filter(file => file.endsWith('.js'));
console.log(commandFiles);

for (const file of commandFiles) {
    const command = require(`../commands/${file}`);
    commands.push(command.data.toJSON());
}

const rest = new REST({ version: '10' }).setToken(token);

// console.log(commands);

(async () => {
    try {
        const rest = new REST({ version: '10' }).setToken(token);
        
        console.log('Started refreshing application (/) commands.');

        //publish to guild if guildID is set, otherwise publish to global
        if (guildID) {
            const data = await rest.put(
                Routes.applicationGuildCommands(clientID, guildID),
                { body: commands },
            );
            console.log('Successfully reloaded '+ data.length +' commands.');
        } else {
            const data = await rest.put(
                Routes.applicationCommands(clientID),
                { body: commands },
            );
            console.log('Successfully reloaded '+ data.length +' commands.');
        }

    } catch (error) {
        console.error(error);
    }
})();

