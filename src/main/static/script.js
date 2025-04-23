$(document).ready(function() {
    let titles = [];
    $.getJSON("/titles", function(data) {
        titles = data;
    });

    $("#movie_title").on("input", function() {
        let input = $(this).val().toLowerCase();
        let matches = titles.filter(t => t.toLowerCase().includes(input)).slice(0, 10);
        let list = matches.map(m => `<option value="${m}">`).join('');
        $("#suggestions").remove();
        $(this).after(`<datalist id="suggestions">${list}</datalist>`);
        $(this).attr("list", "suggestions");
    });
});
