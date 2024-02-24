# SOLID: Cómo hacer un código de calidad

Aunque los 5 principios que agrupa SOLID están especialmente _dirigidos a programación oriendada a objetos_, su uso se extiende mucho más prácticamente para todo código.

Estos principios establecen una serie de prácticas de desarrollo de software con especial **consideración por el mantenimiento del código a medida que este crece**. 
Adoptar estas prácticas contribuye a eludir **code smells** (cuando el código no está limpio), **refactoring code** (restructurar el código) y un desarrollo **Ágil y Adaptativo**.


SOLID representa:

* S - Single-responsiblity Principle :dragon_face:
* O - Open-closed Principle :dolphin:
* L - Liskov Substitution Principle :shark:
* I - Interface Segregation Principle :tiger2:
* D - Dependency Inversion Principle :parrot:

### :dragon_face: Single-Responsibility Principle :dragon_face:
>**Una clase deberái tener un y sólo un trabajo.**

> [!NOTE]
> Esta es la MÁS IMPORTANTE.

### :dolphin: Open-Closed Principle :dolphin:
>**Los objetos o entidades deben estar abiertas para extensión pero cerradas para modificación** .
Es mejor hacer una interfaz robusta antes que abusar de arboles lógicos con los que la función modificaría su funcionamiento.

### :shark: Liskov Substitution Principle :shark:
> **Sea _q(x)_ una propiedad demostrable de los objetos _x_ de tipo _T_. Entonces _q(y)_ debería ser demostrable para objetos _y_ de tipo _S_ donde _S_ es un subtipo de _T_.**
i.e. Toda subclasss debería ser sustituida por su clase madre.

### :tiger2: Interface Segregation Principle :tiger2:
> **Un cliente jamás debería ser forzado a implementar una interficie que no utiliza o a depender de métodos que no utlice.**
Cada clase con sus método únicos e indispensables.

### :parrot: Dependency Inversion Principle :parrot:
>Entidades deben pedender en abstracciones, no en concrepciones. El high-level debede depender en la abstracción de los módulos low-level



>Fuente: https://www.digitalocean.com/community/conceptual-articles/s-o-l-i-d-the-first-five-principles-of-object-oriented-design
