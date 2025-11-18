/* A command that lists all containers with their status */

const { SlashCommandBuilder, EmbedBuilder } = require("discord.js");
const Docker = require('node-docker-api').Docker;

module.exports = {
    data: new SlashCommandBuilder()
        .setName("allcontainers")
        .setDescription("Lists all containers"),
    async execute(interaction) {
        outArray = [];
        interaction.reply('Listing all containers...');

        //create docker client
        const docker = new Docker({ socketPath: '/var/run/docker.sock' });

        // get all containers
        const containers = await docker.container.list({ all: true});

        // create array of containers with name and status
        outArray = containers.map(c => {
            return {
                name: c.data.Names[0].slice(1),
                status: c.data.State
            };
        });

        embedCount = Math.ceil(outArray.length / 25);
            for (let i = 0; i < embedCount; i++) {
                const embed = new EmbedBuilder()
                    .setTitle('Containers')
                    .addFields(outArray.slice(i * 25, (i + 1) * 25).map(e => {
                        return { name: e.name, value: e.status };
                    }))
                    .setColor(0x00AE86);
                interaction.channel.send({ embeds: [embed] });
            }  
    },
};