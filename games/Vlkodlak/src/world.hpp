#include <vector>

#include "raylib.h"

struct World {
    enum class Tile {
        Empty = 0,
        Grass,
        Dirt
    };

    std::vector<Color> tileColors {
        Color { 180, 180, 180, 255 },
        Color { 255, 100, 180, 255 },
        Color { 100, 180, 100, 255 }
    };

    const Vector2 triangleDim{ 2.0f, 1.73205 };
    std::pair<int, int> playerPos{ 0,0 };

    std::vector<std::vector<Tile>> tiles;

    World(const char* worldName);

    void Draw();
};
