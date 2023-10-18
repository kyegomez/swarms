require('dotenv').config();
const fs = require('node:fs');
const path = require('node:path');
const token = process.env.DISCORD_TOKEN;
const clientID = process.env.DISCORD_CLIENT_ID;

// Require the necessary discord.js classes
const { Client, Collection, Events, GatewayIntentBits } = require('discord.js');

// Create a new client instance
const client = new Client({ intents: [GatewayIntentBits.Guilds] });

//run backend/deployCommands.js
const { exec } = require('child_process');
exec('node backend/deployCommands.js', (err, stdout, stderr) => {
    if (err) {
        //some err occurred
        console.error(err);
    } else {
        // print complete output
        console.log(stdout);
    }
});



// When the client is ready, run this code
client.once(Events.ClientReady, c => {
	console.log(`Ready! Logged in as ${c.user.tag}`);
});

// Log in to Discord with your client's token
client.login(token);

// Create a new collection for commands
client.commands = new Collection();

const commandsPath = path.join(__dirname, 'commands');
const commandFiles = fs.readdirSync(commandsPath).filter(file => file.endsWith('.js'));

for (const file of commandFiles) {
    const filePath = path.join(commandsPath, file);
    const command = require(filePath);
    // Set a new item in the Collection with the key as the name of the command and the value as the exported module
    if ('data' in command && 'execute' in command) {
        client.commands.set(command.data.name, command);
    } else {
        console.log(`Command ${file} is missing 'data' or 'execute'`);
    }
}
//build and display invite link
const inviteLink = 'https://discord.com/oauth2/authorize?client_id='+clientID+'&permissions=2147534912&scope=bot%20applications.commands';

console.log(`Invite link: ${inviteLink}`);

// execute on slash command
client.on(Events.InteractionCreate, async interaction => {
    if (interaction.isChatInputCommand()) {
        const command = client.commands.get(interaction.commandName);

        if (!command) {
            console.error('No command matching ${interaction.commandName} was found.');
            return;
        }

        try {
            await command.execute(interaction);
        } catch (error) {
            console.error(error);
            // await interaction.reply({ content: 'There was an error while executing this command!', ephemeral: true });
        }
    } else if (interaction.isAutocomplete()) {

        const command = client.commands.get(interaction.commandName);

        if (!command) {
            console.error('No command matching ${interaction.commandName} was found.');
            return;
        }

        try {
            await command.autocomplete(interaction);
        } catch (error) {
            console.error(error);
            // await interaction.({ content: 'There was an error while executing this command!', ephemeral: true });
        }
    }
});

