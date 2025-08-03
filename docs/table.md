---
layout: page
title: Comparison Data Table
permalink: results
description: "Benchmarking results of different MPPI Implementations on various hardware"
extra_head_content: |
  <!-- PapaParse -->
  <script src="https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js"></script>

  <! -- DataTables w/ ColumnControl -->
  <link href="https://cdn.datatables.net/v/dt/dt-2.3.2/cc-1.0.7/datatables.min.css" rel="stylesheet" integrity="sha384-Cj3XOJTsHbH8wLMuiur9hBHE6VbFJ8lUnGGhQELFs136YAqA3kG8Ljowyz51xPUf" crossorigin="anonymous">
  <script src="https://cdn.datatables.net/v/dt/dt-2.3.2/cc-1.0.7/datatables.min.js" integrity="sha384-uoZRKlUQlPstYKkxPTk3T53KCmifX/+WjwqqN9Q9MMHW1vgL12W9FrIiP/28HpWQ" crossorigin="anonymous"></script>
---
This page provides an interactive index of all the tests done in our [benchmarking]({{ site.url }}{{ site.baseurl }}{% link docs/benchmarks.md %}) of other MPPI implementations.
It covers what implementations we are comparing against, what exactly we are testing, and provides some visualizations of this data.
We may add new hardware comparisons to this page over time.
A PDF of this data can be found [here]({{ site.url }}{{ site.baseurl }}{% link docs/assets/mppi_runtimes_table.pdf %}) and the code used to do these comparisons is available on [GitHub](https://github.com/ACDSLab/MPPI_Paper_Example_Code).

<table id="csv-table" class="display" style="width:100%">
  <thead></thead>
  <tbody></tbody>
</table>

<script>
  Papa.parse('{{ site.url }}{{ site.baseurl }}/assets/data/results.csv', {
    download: true,
    header: true,
    trimHeaders: true,
    complete: function(results) {
      const headers = results.meta.fields;

      const data = results.data.filter(row =>
        headers.some(field => row[field] && row[field].trim() !== '')
      );

      const tableEl = document.getElementById('csv-table');

      // Create header
      const thead = document.querySelector('#csv-table thead');
      thead.innerHTML = '<tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr>';

      // Create body
      const tbody = document.querySelector('#csv-table tbody');
      data.forEach(row => {
        tbody.innerHTML += '<tr>' + headers.map(h => `<td>${row[h]}</td>`).join('') + '</tr>';
      });

      // Init DataTable with SearchPanes
      $('#csv-table').DataTable({
        paging: false,
        columnControl: ['order', ['searchList', 'orderAsc', 'orderDesc', 'orderRemove']],
        order: [[4, 'asc']],
        ordering: {
            indicators: false,
            handler: false
        },
        columnDefs: [
        { // Limit digits of floating point columns
          targets: [4, 5], // zero-based column index
          render: function (data, type, row) {
            const num = parseFloat(data);
            return !isNaN(num) ? num.toFixed(5) : data;
          }
        },
        { // center data in these columns
          targets: [0, 1, 3],
          className: 'dt-center',
        }
      ],
      // Center all header names
      headerCallback: function(thead, data, start, end, display) {
        $(thead).find('th').css('text-align', 'center');
      }
      });
    }
  });
</script>
