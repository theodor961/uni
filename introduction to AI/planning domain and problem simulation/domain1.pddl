(define (domain social_robot)

(:requirements :adl :typing)

(:types
	waypoint 
	robot
)

(:predicates
	(robot_at ?v - robot ?wp - waypoint)
	(connected ?from ?to - waypoint)
	(visited ?wp - waypoint)
        
)



;; Move between any two waypoints, avoiding terrain
(:action goto_waypoint
	:parameters (?v - robot ?from ?to - waypoint)
	
	:precondition (and
		 (robot_at ?v ?from))
	:effect (and
		 (visited ?to)
		 (not (robot_at ?v ?from))
		 (robot_at ?v ?to))
)


