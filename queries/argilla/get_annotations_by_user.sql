SELECT users.first_name, COUNT(responses.id)
FROM responses
    JOIN users ON responses.user_id = users.id
GROUP BY user_id