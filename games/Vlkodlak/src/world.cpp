#include <iostream>

#include "raylib.h"

#include "world.hpp"

World::World(const char* worldName) {
	int dataSize;
	unsigned char* data = LoadFileData(worldName, &dataSize);

	tiles.emplace_back();
	for (int i = 0; i < dataSize; i++) {
		unsigned char c = data[i];

		Tile tile;
		switch (c) {
			case ' ': tile = Tile::Empty; break;
			case 'X': tile = Tile::Grass; break;
			case 'O': tile = Tile::Dirt; break;
			case '\n': tiles.emplace_back(); continue; break;
			default: continue; break;
		}
		tiles[tiles.size() - 1].push_back(tile);
	}

	UnloadFileData(data);
}

void World::Draw() {
	for (size_t i = 0; i < tiles.size(); i++) {
		Vector3 v1{ triangleDim.x * i * 0.5f - triangleDim.x * 0.5f, 1.0f, triangleDim.y * (i + 1) };
		Vector3 v2{ triangleDim.x * i * 0.5f, 1.0f, triangleDim.y * i };
		Vector3 v3{ triangleDim.x * i * 0.5f + triangleDim.x * 0.5f, 1.0f, triangleDim.y * (i + 1) };

		for (size_t j = 0; j < tiles[i].size(); j++) {
			Vector3 tmp = Vector3{ v2.x + triangleDim.x, v2.y, v2.z };
			v1 = v2;
			v2 = v3;
			v3 = tmp;
			if (j % 2 == 0)
				DrawTriangle3D(v1, v2, v3, tileColors[static_cast<int>(tiles[i][j])]);
			else
				DrawTriangle3D(v1, v3, v2, tileColors[static_cast<int>(tiles[i][j])]);
		}
	}
}
