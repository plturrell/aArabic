# Basketball Domain Data Model - Pure Mojo Implementation
# Defines the data structures for the basketball domain

from collections import Dict, List


struct Player:
    """Represents a basketball player."""
    var player_id: String
    var name: String
    var team: String
    var position: String
    var jersey_number: Int
    var height: String
    var weight: Int
    var age: Int
    
    fn __init__(out self, player_id: String, name: String, team: String, 
                position: String, jersey_number: Int):
        self.player_id = player_id
        self.name = name
        self.team = team
        self.position = position
        self.jersey_number = jersey_number
        self.height = ""
        self.weight = 0
        self.age = 0


struct PlayerStats:
    """Represents player statistics."""
    var player_id: String
    var games_played: Int
    var points_per_game: Float64
    var rebounds_per_game: Float64
    var assists_per_game: Float64
    var steals_per_game: Float64
    var blocks_per_game: Float64
    var field_goal_percentage: Float64
    var three_point_percentage: Float64
    var free_throw_percentage: Float64
    
    fn __init__(out self, player_id: String):
        self.player_id = player_id
        self.games_played = 0
        self.points_per_game = 0.0
        self.rebounds_per_game = 0.0
        self.assists_per_game = 0.0
        self.steals_per_game = 0.0
        self.blocks_per_game = 0.0
        self.field_goal_percentage = 0.0
        self.three_point_percentage = 0.0
        self.free_throw_percentage = 0.0


struct Team:
    """Represents a basketball team."""
    var team_id: String
    var name: String
    var city: String
    var conference: String
    var division: String
    var wins: Int
    var losses: Int
    var players: List[String]  # List of player IDs
    
    fn __init__(out self, team_id: String, name: String, city: String):
        self.team_id = team_id
        self.name = name
        self.city = city
        self.conference = ""
        self.division = ""
        self.wins = 0
        self.losses = 0
        self.players = List[String]()
    
    fn win_percentage(self) -> Float64:
        var total = self.wins + self.losses
        if total == 0:
            return 0.0
        return Float64(self.wins) / Float64(total)


struct Game:
    """Represents a basketball game."""
    var game_id: String
    var home_team: String
    var away_team: String
    var home_score: Int
    var away_score: Int
    var date: String
    var status: String  # "scheduled", "in_progress", "final"
    
    fn __init__(out self, game_id: String, home_team: String, away_team: String,
                date: String):
        self.game_id = game_id
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = 0
        self.away_score = 0
        self.date = date
        self.status = "scheduled"
    
    fn winner(self) -> String:
        if self.status != "final":
            return ""
        if self.home_score > self.away_score:
            return self.home_team
        return self.away_team


struct BasketballDB:
    """Database for the basketball domain."""
    var players: Dict[String, Player]
    var player_stats: Dict[String, PlayerStats]
    var teams: Dict[String, Team]
    var games: Dict[String, Game]
    
    fn __init__(out self):
        self.players = Dict[String, Player]()
        self.player_stats = Dict[String, PlayerStats]()
        self.teams = Dict[String, Team]()
        self.games = Dict[String, Game]()
    
    fn add_player(inout self, player: Player):
        self.players[player.player_id] = player
    
    fn add_team(inout self, team: Team):
        self.teams[team.team_id] = team
    
    fn add_game(inout self, game: Game):
        self.games[game.game_id] = game
    
    fn get_player(self, player_id: String) raises -> Player:
        if player_id not in self.players:
            raise Error("Player " + player_id + " not found")
        return self.players[player_id]
    
    fn get_team(self, team_id: String) raises -> Team:
        if team_id not in self.teams:
            raise Error("Team " + team_id + " not found")
        return self.teams[team_id]

