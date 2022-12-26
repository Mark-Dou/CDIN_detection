from utils.plugin_loader import PluginLoader

if __name__ == '__main__':
    trainer = PluginLoader.get_trainer('trainer')()
    trainer.run()
