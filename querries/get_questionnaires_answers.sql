SELECT au.username, qa.updated_at,
       qqre.name, qq.question_text,
       qa.answer_text

FROM auth_user au
    JOIN questionnaires_answer qa ON au.id = qa.user_id
    JOIN questionnaires_question qq ON qa.question_id = qq.id
    JOIN questionnaires_questionnaire qqre ON qq.questionnaire_id = qqre.id

ORDER BY username, updated_at