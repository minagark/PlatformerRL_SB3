class Entity():
    def __init__(self, x, y, width, height, color, gravity) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.gravity = gravity

    def collision_type(self, other):
        collides = False
        dist_x = other.x - self.x
        dist_y = other.y - self.y

        if 0 < dist_y <= self.height and -other.width < dist_x < self.width:
            collides = "bottom"
        elif -other.height <= dist_y < 0 and -other.width < dist_x < self.width:
            collides = "top"
        elif -other.height < dist_y < self.height and 0 < dist_x <= self.width:
            collides = "right"
        elif -other.height < dist_y < self.height and -other.width <= dist_x < 0:
            collides = "left"
        else:
            collides = False
        
        return collides

class Player(Entity):
    def __init__(self, x, y, width, height, color, gravity, jump_height) -> None:
        super().__init__(x, y, width, height, color, gravity)
        self.jump_height = jump_height

        self.vx = 0
        self.vy = 0
        self.max_vx = 4
        self.max_vy = 45
        self.ax = 0
        self.ay = 0
        self.pressing_right = False
        self.pressing_left = False
        self.pressing_up = False
        self.pressing_down = False
        self.on_ground = False

    # def update():
    #     pass

class Platform(Entity):
    def __init__(self, x, y, width, height, color, platform_type) -> None:
        super().__init__(x, y, width, height, color, 0)
        self.platform_type = platform_type
        #bouncy, (purple) sticky, phasing (partially transparent), smth else; make this a String