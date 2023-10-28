SELECT au.username, qa.updated_at,
       qqre.name, qq.question_text,
       qa.answer_text

FROM auth_user au
    JOIN questionnaires_answer qa ON au.id = qa.user_id
    JOIN questionnaires_question qq ON qa.question_id = qq.id
    JOIN questionnaires_questionnaire qqre ON qq.questionnaire_id = qqre.id

WHERE au.username IN ('1120', '1160', '1210', '1211', '1215', '1257', '1314', '1436', '1638',
                      '1847', '2238', '2290', '2621', '3014', '3163', '3465', '3758', '4060',
                      '4073', '4384', '4411', '4444', '4482', '4520', '4643', '4760', '4963',
                      '5282', '5362', '5418', '5529', '5955', '6353', '6513', '6556', '6734',
                      '6751', '6818', '6954', '6992', '7102', '7597', '7621', '7667', '7691',
                      '7727', '8194', '8241', '8342', '8533', '8583', '8858', '8889', '9178',
                      '9224', '9327', '9458', '9672', '2525', '3884', '9707', '4339', '1640')

ORDER BY username, updated_at