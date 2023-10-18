const { SlashCommandBuilder, EmbedBuilder } = require("discord.js");
const Docker = require('node-docker-api').Docker;

module.exports = {
    data: new SlashCommandBuilder()
        .setName("startcontainer")
        .setDescription("Starts a Docker container")
        .addStringOption(option =>
            option.setName('container')
                .setDescription('The container to start')
                .setRequired(true)
                .setAutocomplete(true)),
    async autocomplete(interaction) {
        try {
            // Create docker client
            const docker = new Docker({ socketPath: '/var/run/docker.sock' });

            // Get list of running containers
            const containers = await docker.container.list({ all: true, filters: { status: ['exited'] } });
            const runningContainers = containers.map(c => c.data.Names[0].slice(1));

            // Filter list of containers by focused value
            const focusedValue = interaction.options.getFocused(true);
            const filteredContainers = runningContainers.filter(container => container.startsWith(focusedValue.value));

            //slice if more than 25
            let sliced;
            if (filteredContainers.length > 25) {
                sliced = filteredContainers.slice(0, 25);
            } else {
                sliced = filteredContainers;
            }

            // Respond with filtered list of containers
            await interaction.respond(sliced.map(container => ({ name: container, value: container })));

        } catch (error) {
            // Handle error
            console.error(error);
            await interaction.reply('An error occurred while getting the list of running containers.');
        }
    },
    async execute(interaction) {
        try {
            // Get container name from options
            const containerName = interaction.options.getString('container');

            // Start container in interactive mode
            await interaction.reply(`Starting container "${containerName}" in interactive mode...`);
            const container = docker.getContainer(containerName);
            const info = await container.inspect();
            if (!info) {
                await interaction.followUp(`Container "${containerName}" does not exist.`);
                throw new Error(`Container "${containerName}" does not exist.`);
            }
            await container.start({
                AttachStdin: true,
                AttachStdout: true,
                AttachStderr: true,
                Tty: true,
                OpenStdin: true,
                StdinOnce: false
            });

            // Attach to container's streams
            const stream = await container.attach({
                stream: true,
                stdin: true,
                stdout: true,
                stderr: true
            });

            // Use socket.io for real-time communication with the container
            io.on('connection', (socket) => {
                socket.on('containerInput', (data) => {
                    stream.write(data + '\n'); // Send input to the container
                });

                stream.on('data', (data) => {
                    socket.emit('containerOutput', data.toString()); // Send container's output to the client
                });
            });

            // Confirm that container was started
            await interaction.followUp(`Container "${containerName}" was successfully started in interactive mode.`);
        } catch (error) {
            // Handle error
            console.error(error);
            await interaction.followUp(`An error occurred while trying to start the container "${containerName}" in interactive mode.`);
        }
    },
};
