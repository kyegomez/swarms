const { SlashCommandBuilder, EmbedBuilder } = require("discord.js");
const Docker = require('node-docker-api').Docker;

module.exports = {
  data: new SlashCommandBuilder()
    .setName("stopcontainer")
    .setDescription("Stops a Docker container")
    .addStringOption(option => 
      option.setName('container')
        .setDescription('The container to stop')
        .setRequired(true)
        .setAutocomplete(true)),
  async autocomplete(interaction) {
    try {
      // Create docker client
      const docker = new Docker({ socketPath: '/var/run/docker.sock' });

      // Get list of running containers
      const containers = await docker.container.list({ all: true, filters: { status: ['running'] } });
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
      // create docker client
      const docker = new Docker({ socketPath: '/var/run/docker.sock' });

      // Get container name from options
      const container = interaction.options.getString('container');
      
      // Restart container
      await interaction.reply(`Stopping container "${container}"...`);
      const containers = await docker.container.list({ all: true, filters: { name: [container] } });
          if (containers.length === 0) {
            await interaction.followUp(`Container "${container}" does not exist.`);
            throw new Error(`Container "${container}" does not exist.`);
          }
          await containers[0].stop();

      // Confirm that container was restarted
      await interaction.followUp(`Container "${container}" was successfully stopped.`);
    } catch (error) {
      // Handle error
      console.error(error);
      await interaction.followUp(`An error occurred while trying to stop the container "${container}".`);
    }
  }
};
