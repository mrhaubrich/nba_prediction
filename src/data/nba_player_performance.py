# Player;Pos;Age;Tm;G;GS;MP;FG;FGA;FG%;3P;3PA;3P%;2P;2PA;2P%;eFG%;FT;FTA;FT%;ORB;DRB;TRB;AST;STL;BLK;TOV;PF;PTS;Performance
class NBAPlayerPerformance:
    player: str
    position: str
    age: int
    team: str
    games: int
    games_started: int
    minutes_played: int
    field_goals: int
    field_goal_attempts: int
    field_goal_percentage: float
    three_point_field_goals: int
    three_point_field_goal_attempts: int
    three_point_field_goal_percentage: float
    two_point_field_goals: int
    two_point_field_goal_attempts: int
    two_point_field_goal_percentage: float
    effective_field_goal_percentage: float
    free_throws: int
    free_throw_attempts: int
    free_throw_percentage: float
    offensive_rebounds: int
    defensive_rebounds: int
    total_rebounds: int
    assists: int
    steals: int
    blocks: int
    turnovers: int
    personal_fouls: int
    points: int
    performance: float
