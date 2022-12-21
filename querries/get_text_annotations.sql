SELECT *
FROM labels_scale

-- SELECT au.username, lt.updated_at, lt.question, lt.text

-- FROM examples_example
-- FROM labels_scale
-- FROM labels_category -- empty table
-- FROM labels_textlabel
-- FROM label_types_scaletype

-- FROM auth_user au JOIN labels_textlabel lt ON au.id = lt.user_id

-- ORDER BY au.username, lt.updated_at

-- SELECT auth_user.username, questionnaires_answer.updated_at,
--        questionnaires_questionnaire.name, questionnaires_question.question_text,
--        questionnaires_answer.answer_text

-- FROM auth_user JOIN questionnaires_answer
--         ON auth_user.id = questionnaires_answer.user_id
--     JOIN questionnaires_question
--         ON questionnaires_answer.question_id = questionnaires_question.id
--     JOIN questionnaires_questionnaire
--         ON questionnaires_question.questionnaire_id = questionnaires_questionnaire.id

-- WHERE auth_user.username = '6170'

-- ORDER BY username, updated_at