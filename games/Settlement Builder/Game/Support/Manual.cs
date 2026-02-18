using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Game.Support
{
    /// <summary>
    /// Implements game manual.
    /// </summary>
    internal class Manual
    {
        internal Dictionary<string, string> manual = new Dictionary<string, string>();

        // Add manuals entry for each clickable element in game
        internal Manual()
        {
            // Basic tiles
            manual.Add("TileEmpty", "Empty tile\n" +
                "This tile does not provide any resources.");
            manual.Add("TileWood", "Forrest\n" +
                "Forrest provides the Wood resource.");
            manual.Add("TileStone", "Rocks\n" +
                "Rocks provide the Stone resource.");
            manual.Add("TileFood", "Wild boar herd\n" +
                "Wild boars provide the Food resource.");
            manual.Add("TileIron", "Iron vein\n" +
                "Rock with an iron vein provides the Iron resource.");

            // Resource items
            manual.Add("ResourceItemWood", "Wood resource\n" +
                "Wood can be obtained by building Sawmills in Forrests.");
            manual.Add("ResourceItemStone", "Stone resource\n" +
                "Stone can be obtained by building Quarries on Rocks.");
            manual.Add("ResourceItemFood", "Food resource\n" +
                "Food can be obtained by hunting Wild boars from Camps or from Windmills.");
            manual.Add("ResourceItemIron", "Iron resource\n" +
                "Iron can be obtained by building a Mine on an Iron vein rock.");
            manual.Add("ResourceItemGold", "Gold resource\n" +
                "Gold can be obtained by forging weapons in a Forge and collecting taxes from Huts.");

            // Building tiles
            manual.Add("TileSawmill", "Sawmill\n" +
                "Sawmill cuts down trees and generates Wood resource.\n" +
                "Cost to build: 5 Wood     Cost per day: None     Production per day: 1 Wood");
            manual.Add("TileLumber", "Lumber\n" +
                "Improved Sawmill generates Wood resource.\n" +
                "Cost to build: 10 Wood, 1 Food     Cost per day: 1 Food     Production per day: 3 Wood");
            manual.Add("TileCamp", "Camp\n" +
                "Hunting camps hunt down wild boars and generate Food resource.\n" +
                "Cost to build: 8 Wood     Cost per day: None     Production per day: 2 Food");
            manual.Add("TileQuarry", "Quarry\n" +
                "Quarry mines rocks and generates Stone resource.\n" +
                "Cost to build: 30 Wood, 8 Food, 5 Gold     Cost per day: 5 Food     Production per day: 1 Stone");
            manual.Add("TileHut", "Hut\n" +
                "Hut houses peasants who pay taxes and thus generate Gold resource.\n" +
                "Cost to build: 20 Wood     Cost per day: 2 Food     Production per day: 1 Gold");
            manual.Add("TileWindmill", "Windmill\n" +
                "Windmill generates Food resource.\n" +
                "Cost to build: 25 Wood, 1 Stone, 5 Gold     Cost per day: 1 Gold     Production per day: 1 Food");
            manual.Add("TileTower", "Tower\n" +
                "Tower generates Food resource.\n" +
                "Cost to build: 25 Wood, 15 Stone     Cost per day: 5 Wood     Production per day: 5 Food");
            manual.Add("TileMine", "Iron mine\n" +
                "Mine generates Iron resource. Building this wins the game!\n" +
                "Cost to build: 20 Wood, 20 Stone, 20 Food, 20 Gold     Cost per day: None     Production per day: 1 Iron");
        }

        // Return manual entry
        internal string GetText(string type)
        {
            return manual[type];
        }
    }
}
