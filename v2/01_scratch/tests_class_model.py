class marca_cl():
    def __init__(self, models, colors):
        self.models = models
        self.colors = colors
        
    def get_models(self):
        print(self.models)
        
    def get_colors(self):
        print(self.colors)

    def get_extended(self):
        def extend_colors_f(colors):
            aux = [x + '_kuku' for x in colors]
            return aux
        
        extended = extend_colors_f(self.colors)
        print(extended)

marca1 = marca_cl(['escort', 'fiesta'], ['rojo', 'blanco'])
marca1.get_extended()

#%%

class marca_cl():
    def __init__(self, models, colors):
        self.models = models
        self.colors = colors
        
    def get_models(self):
        print(self.models)
        
    def get_colors(self):
        print(self.colors)

    def extend_colors_f(self, colors):
        aux = [x + '_kuku' for x in colors]
        return aux

    def get_extended(self):
        extended = self.extend_colors_f(self.colors)
        print(extended)

marca1 = marca_cl(['escort', 'fiesta'], ['rojo', 'blanco'])
marca1.get_extended()

#%%